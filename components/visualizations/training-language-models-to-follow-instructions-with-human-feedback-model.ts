// A pure, deterministic model of the RLHF step from "Training language models to
// follow instructions with human feedback" (InstructGPT). The reinforcement
// learning stage maximizes the paper's objective (equation 2):
//
//   objective(phi) = E_pi[ r_theta(x, y) ]
//                  - beta * KL( pi^RL(y|x) || pi^SFT(y|x) )
//                  + gamma * E_pretrain[ log pi^RL(x) ]
//
// where pi^RL is the learned policy, pi^SFT the supervised baseline it started
// from, beta the KL reward coefficient, and gamma the pretraining-mix coefficient
// (gamma = 0 recovers plain "PPO"; gamma > 0 is "PPO-ptx").
//
// We collapse the response space to a handful of named response "styles" so the
// whole loop is legible. The policy is a tabular softmax over styles: one
// preference theta per style, pi = softmax(theta). Each style carries three
// numbers the RL stage never sees directly:
//   - trueQuality: the hidden human preference for that style (the thing we
//     actually want to maximize, but cannot measure during RL).
//   - rmScore:     what the reward model reports for that style. A good reward
//     model has rmScore close to trueQuality; corrupting a style's rmScore above
//     its trueQuality is exactly the over-optimization the paper warns about.
//   - capability:  how much that style preserves the base model's NLP ability,
//     which the pretraining-mix (ptx) term protects. Pure PPO erodes it (the
//     "alignment tax"); ptx holds it up.
//
// Each tick is one gradient-ascent update on the objective above. For a tabular
// softmax the gradient is closed form and needs no sampling:
//
//   d objective / d theta_i = pi_i * ( advantage_i - sum_j pi_j * advantage_j )
//   advantage_i = rmScore_i - beta * (log(pi_i / piSft_i) + 1) + gamma * capability_i
//
// The bracket subtracts the policy-mean (a baseline), which is what keeps the
// softmax gradient a proper ascent direction. The model also tracks the true
// objective E_pi[trueQuality] so the view can show proxy reward climbing while
// true quality peaks and falls under reward hacking. All randomness comes from a
// seeded stream threaded through state, so a seed plus a list of inputs
// reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

export interface Style {
  name: string;
  blurb: string;
  trueQuality: number;
  rmScore: number;
  capability: number;
}

// The five response styles the policy chooses among. trueQuality is the human
// preference we want; rmScore is the reward model's (imperfect) read of it.
// "Helpful" is the genuinely good answer. "Sycophantic" tells the user what they
// want to hear, which a reward model over-rates because labelers reward agreeable
// answers, so it is the natural reward-hacking target. "Hedging" over-qualifies
// every answer. "Toxic" is confident but harmful. "Terse" is correct but curt.
export const BASE_STYLES: Style[] = [
  {
    name: "Helpful",
    blurb:
      "Follows the instruction, stays honest, gives a useful answer. The behavior we actually want, and what the reward model rewards.",
    trueQuality: 0.92,
    rmScore: 0.9,
    capability: 0.6,
  },
  {
    name: "Sycophantic",
    blurb:
      "Tells the user what they want to hear. A reward model trained on agreeable rankings tends to over-rate this, so it is the natural reward-hacking target.",
    trueQuality: 0.45,
    rmScore: 0.62,
    capability: 0.5,
  },
  {
    name: "Hedging",
    blurb:
      "Long, over-qualified answers that rarely commit. The paper notes InstructGPT learns to hedge because labelers reward epistemic humility.",
    trueQuality: 0.5,
    rmScore: 0.55,
    capability: 0.62,
  },
  {
    name: "Toxic",
    blurb:
      "Confident but harmful or biased. Low true quality, and a calibrated reward model scores it low too.",
    trueQuality: 0.08,
    rmScore: 0.12,
    capability: 0.45,
  },
  {
    name: "Terse",
    blurb:
      "Short, plain answers that keep close to the base model, so they hold the most NLP-benchmark skill. Decent but not the best on reward; the pretraining mix (ptx) leans on these to pay down the alignment tax.",
    trueQuality: 0.7,
    rmScore: 0.78,
    capability: 0.95,
  },
];

export const N_STYLES = BASE_STYLES.length;

export function softmax(theta: number[]): number[] {
  const max = Math.max(...theta);
  const exps = theta.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

function klDivergence(pi: number[], ref: number[]): number {
  let kl = 0;
  for (let i = 0; i < pi.length; i++) {
    const p = at(pi, i);
    if (p > 1e-9) kl += p * Math.log(p / Math.max(at(ref, i), 1e-9));
  }
  return kl;
}

function expected(pi: number[], values: number[]): number {
  let sum = 0;
  for (let i = 0; i < pi.length; i++) sum += at(pi, i) * at(values, i);
  return sum;
}

export interface RlParams {
  beta: number;
  gamma: number;
  learningRate: number;
  rmBias: number[];
  seed: number;
  noise: number;
}

export interface SimState {
  theta: number[];
  thetaSft: number[];
  step: number;
  rngState: number;
  params: RlParams;
  log: EventLog<Snapshot>;
}

// A logged moment carries everything needed to replay the run from here: the
// preferences, the SFT reference, the parameter set, the step counter, and the
// rng cursor.
export interface Snapshot {
  theta: number[];
  thetaSft: number[];
  step: number;
  rngState: number;
  params: RlParams;
}

export interface Readout {
  pi: number[];
  piSft: number[];
  rmScores: number[];
  proxyReward: number;
  trueQuality: number;
  baselineTrueQuality: number;
  kl: number;
  capability: number;
  baselineCapability: number;
  styles: Style[];
}

// The reward model's reported score for each style: its calibrated score plus
// whatever bias the reader has injected into that style. Clamped to [0, 1.4] so a
// corrupted style can out-score the honest one without running away.
function effectiveRmScores(params: RlParams): number[] {
  return BASE_STYLES.map((style, i) =>
    Math.min(1.4, Math.max(0, style.rmScore + at(params.rmBias, i))),
  );
}

export function readout(state: SimState): Readout {
  const pi = softmax(state.theta);
  const piSft = softmax(state.thetaSft);
  const rmScores = effectiveRmScores(state.params);
  const trueValues = BASE_STYLES.map((style) => style.trueQuality);
  const capValues = BASE_STYLES.map((style) => style.capability);
  return {
    pi,
    piSft,
    rmScores,
    proxyReward: expected(pi, rmScores),
    trueQuality: expected(pi, trueValues),
    baselineTrueQuality: expected(piSft, trueValues),
    kl: klDivergence(pi, piSft),
    capability: expected(pi, capValues),
    baselineCapability: expected(piSft, capValues),
    styles: BASE_STYLES,
  };
}

// One PPO update: closed-form gradient ascent on equation 2 over the tabular
// softmax. Optional reproducible noise stands in for the variance of a real PPO
// estimate; with noise 0 the update is exact.
function ascend(state: SimState): SimState {
  const pi = softmax(state.theta);
  const piSft = softmax(state.thetaSft);
  const rmScores = effectiveRmScores(state.params);
  const { beta, gamma, learningRate } = state.params;

  let rngState = state.rngState;
  const advantages = BASE_STYLES.map((style, i) => {
    const klTerm =
      beta * (Math.log(at(pi, i) / Math.max(at(piSft, i), 1e-9)) + 1);
    const capTerm = gamma * style.capability;
    let value = at(rmScores, i) - klTerm + capTerm;
    if (state.params.noise > 0) {
      const draw = nextRandom(rngState);
      rngState = draw.state;
      value += (draw.value - 0.5) * state.params.noise;
    }
    return value;
  });

  const baseline = expected(pi, advantages);
  const theta = state.theta.map(
    (value, i) =>
      value + learningRate * at(pi, i) * (at(advantages, i) - baseline),
  );

  return { ...state, theta, rngState, step: state.step + 1 };
}

export type Intervention =
  | { type: "setBeta"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setLearningRate"; value: number }
  | { type: "setNoise"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "corruptReward"; index: number; amount: number }
  | { type: "nudgeStyle"; index: number; amount: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: RlParams;
}

// The SFT starting point: a model already nudged toward helpfulness by
// supervised fine-tuning, but still spread across the other styles. Both the live
// policy and the fixed KL reference start here.
const SFT_THETA = [0.8, 0.2, 0.3, -0.4, 0.1];

function zeroBias(): number[] {
  return new Array<number>(N_STYLES).fill(0);
}

export const PRESETS: Preset[] = [
  {
    id: "aligned",
    label: "Aligned (honest reward)",
    blurb:
      "A calibrated reward model and a real KL leash. Press play and the policy climbs toward Helpful, and true quality climbs with proxy reward because the reward model isn't lying.",
    params: {
      beta: 0.2,
      gamma: 0,
      learningRate: 1.2,
      rmBias: zeroBias(),
      seed: 7,
      noise: 0,
    },
  },
  {
    id: "hacking",
    label: "Reward hacking",
    blurb:
      "The reward model over-rates Sycophantic and the KL leash is almost off. Play it and watch proxy reward keep climbing while true quality peaks and falls, because the policy chases the cheat.",
    params: {
      beta: 0.01,
      gamma: 0,
      learningRate: 1.4,
      rmBias: [0, 0.55, 0, 0, 0],
      seed: 7,
      noise: 0,
    },
  },
  {
    id: "tax",
    label: "Alignment tax (PPO vs ptx)",
    blurb:
      "Plain PPO with gamma off. The policy gains reward but capability sags as it drifts off the styles that kept old skills. Turn the pretraining mix up and capability comes back.",
    params: {
      beta: 0.08,
      gamma: 0,
      learningRate: 1.2,
      rmBias: zeroBias(),
      seed: 7,
      noise: 0,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "aligned"): SimState {
  const preset = getPreset(presetId);
  return {
    theta: [...SFT_THETA],
    thetaSft: [...SFT_THETA],
    step: 0,
    rngState: preset.params.seed >>> 0,
    params: { ...preset.params, rmBias: [...preset.params.rmBias] },
    log: emptyLog(),
  };
}

function snapshotOf(state: SimState): Snapshot {
  return {
    theta: [...state.theta],
    thetaSft: [...state.thetaSft],
    step: state.step,
    rngState: state.rngState,
    params: { ...state.params, rmBias: [...state.params.rmBias] },
  };
}

function logged(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.step, label, snapshotOf(state)),
  };
}

function styleName(index: number): string {
  return BASE_STYLES[index]?.name ?? `#${index}`;
}

export type RlEvent = SimEvent<Snapshot>;

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setBeta":
      return logged(
        { ...state, params: { ...state.params, beta: action.value } },
        `KL beta -> ${action.value.toFixed(2)}`,
      );
    case "setGamma":
      return logged(
        { ...state, params: { ...state.params, gamma: action.value } },
        `ptx gamma -> ${action.value.toFixed(2)}`,
      );
    case "setLearningRate":
      return logged(
        { ...state, params: { ...state.params, learningRate: action.value } },
        `step size -> ${action.value.toFixed(2)}`,
      );
    case "setNoise":
      return logged(
        { ...state, params: { ...state.params, noise: action.value } },
        `RL noise -> ${action.value.toFixed(2)}`,
      );
    case "setSeed":
      return logged(
        {
          ...state,
          params: { ...state.params, seed: action.value },
          rngState: action.value >>> 0,
        },
        `seed -> ${action.value}`,
      );
    case "corruptReward": {
      const rmBias = [...state.params.rmBias];
      rmBias[action.index] = (rmBias[action.index] ?? 0) + action.amount;
      return logged(
        { ...state, params: { ...state.params, rmBias } },
        `corrupt RM on ${styleName(action.index)} ${action.amount > 0 ? "+" : ""}${action.amount.toFixed(2)}`,
      );
    }
    case "nudgeStyle": {
      const theta = [...state.theta];
      theta[action.index] = (theta[action.index] ?? 0) + action.amount;
      return logged(
        { ...state, theta },
        `nudge ${styleName(action.index)} ${action.amount > 0 ? "+" : ""}${action.amount.toFixed(1)}`,
      );
    }
    case "loadPreset": {
      const next = initialState(action.id);
      return logged(next, `preset -> ${getPreset(action.id).label}`);
    }
    case "restore":
      return logged(
        {
          ...state,
          theta: [...action.snapshot.theta],
          thetaSft: [...action.snapshot.thetaSft],
          step: action.snapshot.step,
          rngState: action.snapshot.rngState,
          params: {
            ...action.snapshot.params,
            rmBias: [...action.snapshot.params.rmBias],
          },
        },
        action.label,
      );
  }
}

// The pure reducer. A tick runs one PPO update; an intervention changes a
// parameter or the policy and records what happened. Every step is deterministic
// given the state, which carries its own rng cursor.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return ascend(state);
  return applyIntervention(state, input.action);
}
