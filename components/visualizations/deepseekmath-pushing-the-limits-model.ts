// A pure, deterministic model of Group Relative Policy Optimization (GRPO) from
// "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
// Models". The math is the paper's: for one question, sample a group of G
// answers from the current policy, score each with a reward model, turn those
// rewards into advantages by subtracting the group mean and dividing by the
// group standard deviation (Â_i = (r_i - mean(r)) / std(r)), then take a clipped
// PPO-style step on the policy with a KL term pulling back toward a frozen
// reference. The group itself is the baseline, so there is no separate value
// model.
//
// The policy here is a softmax over a handful of answer kinds on a single
// question, not a transformer. That keeps the group-relative advantage, the
// clip, and the KL term visible and pokeable while staying honest about being a
// toy. No Math.random reaches the math; all sampling draws from a seeded RNG
// carried in the state, so a seed plus a list of inputs reproduces a run exactly.

export type Rng = () => number;

// mulberry32: a small, fast, seedable PRNG. Same seed gives the same stream, so
// the whole simulation replays deterministically.
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new Error(`index ${index} out of range for length ${items.length}`);
  }
  return value;
}

// An answer kind the policy can produce on this one question. trueReward is what
// a perfect grader would give it. Several kinds can be "correct" so the model can
// show Pass@K coverage holding flat while Maj@K sharpens onto one favored kind.
export interface AnswerKind {
  id: string;
  label: string;
  correct: boolean;
  trueReward: number;
}

export const ANSWER_KINDS: AnswerKind[] = [
  {
    id: "right-clean",
    label: "right, clean steps",
    correct: true,
    trueReward: 1,
  },
  {
    id: "right-messy",
    label: "right, messy steps",
    correct: true,
    trueReward: 1,
  },
  { id: "wrong-close", label: "wrong, close", correct: false, trueReward: 0 },
  { id: "wrong-off", label: "wrong, off track", correct: false, trueReward: 0 },
  { id: "gibberish", label: "gibberish", correct: false, trueReward: 0 },
];

export const KIND_COUNT = ANSWER_KINDS.length;

export interface GrpoParams {
  groupSize: number; // G, how many answers sampled per question
  klCoef: number; // beta, strength of the pull back to the reference policy
  clipEps: number; // epsilon, the PPO clip width
  learningRate: number; // step size on the logits
  rewardNoise: number; // std of gaussian noise the reward model adds to a score
  normalizeStd: boolean; // divide by group std, or just subtract the mean
  hackKind: number; // an answer kind the reward model over-rewards, or -1 for none
  hackBonus: number; // extra reward the model hands the hacked kind
}

// One sampled answer inside a group, with everything the reader needs to see why
// it got the advantage it did.
export interface Sample {
  kindIndex: number;
  reward: number; // what the reward model gave it (may be noisy or hacked)
  advantage: number; // group-relative, Â_i
  ratio: number; // pi_theta / pi_old for its kind, the PPO ratio
  clipped: boolean; // did the clip cap this sample's contribution
}

export interface StepStats {
  groupMean: number;
  groupStd: number;
  klToRef: number; // D_KL(pi_theta || pi_ref), Schulman's unbiased estimator
  clippedFraction: number;
}

export interface SimState {
  params: GrpoParams;
  seed: number;
  rngCursor: number; // advances every draw so the stream never repeats within a run
  logits: number[]; // current policy, one logit per answer kind
  refLogits: number[]; // frozen reference policy (the SFT model)
  oldLogits: number[]; // pi_old, the policy the group was sampled from this step
  step: number;
  lastGroup: Sample[];
  lastStats: StepStats;
  history: HistoryPoint[]; // per-step emergent quantities, for the chart
  log: LogEntry[];
}

export interface HistoryPoint {
  step: number;
  pCorrect: number; // total probability mass on correct kinds (drives Maj@K)
  pTop: number; // probability of the single most likely kind
  passAtK: number; // chance at least one of K samples is correct (coverage)
  majAtK: number; // chance the majority vote of K samples is correct
  kl: number;
}

export interface LogEntry {
  id: number;
  step: number;
  label: string;
  snapshot: SimState;
}

export type Intervention =
  | { type: "setGroupSize"; value: number }
  | { type: "setKlCoef"; value: number }
  | { type: "setClipEps"; value: number }
  | { type: "setLearningRate"; value: number }
  | { type: "setRewardNoise"; value: number }
  | { type: "toggleNormalizeStd" }
  | { type: "setHack"; kindIndex: number }
  | { type: "setHackBonus"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: SimState; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

// Draw one answer kind from a probability vector using the seeded stream.
function sampleKind(probs: number[], rng: Rng): number {
  const roll = rng();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += at(probs, i);
    if (roll < cumulative) return i;
  }
  return probs.length - 1;
}

// Box-Muller, fed from the seeded stream, so the reward noise is reproducible.
function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// The reward model's score for one answer kind: its true reward, plus optional
// noise, plus the hack bonus if the reader pointed the model at a kind to
// over-reward. This is where reward hacking enters; a wrong kind can be handed a
// high score and GRPO will faithfully chase it.
function scoreKind(kindIndex: number, params: GrpoParams, rng: Rng): number {
  const base = at(ANSWER_KINDS, kindIndex).trueReward;
  const noise = params.rewardNoise > 0 ? gaussian(rng) * params.rewardNoise : 0;
  const hack = kindIndex === params.hackKind ? params.hackBonus : 0;
  return base + noise + hack;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values: number[], avg: number): number {
  if (values.length === 0) return 0;
  const variance =
    values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

// Probability that the majority vote over K independent samples is a correct
// kind. The favored correct kind wins the vote whenever it is the plurality, so
// this tracks pTop closely once the policy sharpens onto a correct answer, which
// is exactly the Maj@K curve the paper reports rising.
function majorityCorrectProb(probs: number[], k: number): number {
  // Closed form over K is expensive; estimate with the dominant-kind heuristic
  // the paper's plots follow: the vote lands on the single most likely kind, and
  // it is correct when that kind is correct. Mass on correct kinds beyond the top
  // one adds a small lift. This stays monotonic in pCorrect and pTop, which is
  // what the visualization needs.
  const topIndex = probs.indexOf(Math.max(...probs));
  const topCorrect = at(ANSWER_KINDS, topIndex).correct;
  const pTop = at(probs, topIndex);
  const sharpness = 1 - Math.pow(1 - pTop, k); // more votes, cleaner plurality
  const pCorrect = probs.reduce(
    (sum, p, i) => sum + (at(ANSWER_KINDS, i).correct ? p : 0),
    0,
  );
  if (topCorrect) return pCorrect * sharpness + pCorrect * (1 - sharpness);
  return pCorrect * (1 - sharpness);
}

// Probability that at least one of K samples is a correct kind. This is the
// coverage the paper finds RL does not improve: it depends only on how much mass
// sits on correct kinds, and the GRPO step moves mass between kinds without
// inventing a new correct kind, so pass@K barely moves.
function passAtKProb(probs: number[], k: number): number {
  const pWrong = probs.reduce(
    (sum, p, i) => sum + (at(ANSWER_KINDS, i).correct ? 0 : p),
    0,
  );
  return 1 - Math.pow(pWrong, k);
}

const REPORT_K = 16;

function snapshotPoint(state: SimState, kl: number): HistoryPoint {
  const probs = softmax(state.logits);
  const pCorrect = probs.reduce(
    (sum, p, i) => sum + (at(ANSWER_KINDS, i).correct ? p : 0),
    0,
  );
  return {
    step: state.step,
    pCorrect,
    pTop: Math.max(...probs),
    passAtK: passAtKProb(probs, REPORT_K),
    majAtK: majorityCorrectProb(probs, REPORT_K),
    kl,
  };
}

// One GRPO training step. Sample a group from the current policy (which becomes
// pi_old for this step), score each answer, normalize the rewards into
// group-relative advantages, then take a clipped surrogate step with a KL term.
// Returns the next state; pure given the seeded cursor in state.
function grpoStep(state: SimState): SimState {
  const { params } = state;
  const rng = mulberry32((state.seed + state.rngCursor * 2654435761) >>> 0);

  const oldProbs = softmax(state.logits);
  const groupSize = Math.max(2, Math.round(params.groupSize));

  const kinds: number[] = [];
  const rewards: number[] = [];
  for (let i = 0; i < groupSize; i++) {
    const kind = sampleKind(oldProbs, rng);
    kinds.push(kind);
    rewards.push(scoreKind(kind, params, rng));
  }

  const groupMean = mean(rewards);
  const groupStd = std(rewards, groupMean);
  const denom = params.normalizeStd ? Math.max(groupStd, 1e-4) : 1;

  // Advantage per sample, Â_i. Every token in an output shares the output's
  // advantage under outcome supervision, so one number per sample is faithful.
  const advantages = rewards.map((reward) => (reward - groupMean) / denom);

  // Accumulate the GRPO gradient on the logits. For a softmax policy the score
  // function for kind c is (1[k==c] - pi(c)). The clipped surrogate caps the
  // ratio's contribution to within [1-eps, 1+eps] when it would push past the
  // trust region, which is the min() in equation 3. pi_theta == pi_old at the
  // start of the step, so the ratio is 1 and nothing clips on the first micro-
  // step; the clip still bounds how far one big advantage can drag the update.
  const grad = new Array<number>(KIND_COUNT).fill(0);
  let clippedCount = 0;
  const samples: Sample[] = kinds.map((kind, i) => {
    const advantage = at(advantages, i);
    const ratio = 1; // single inner update, pi_theta starts equal to pi_old
    const clipLow = 1 - params.clipEps;
    const clipHigh = 1 + params.clipEps;
    const clippedRatio = Math.min(Math.max(ratio, clipLow), clipHigh);
    // The PPO objective takes the smaller of the two surrogates. With ratio == 1
    // they agree, but a positive advantage that would later push the ratio above
    // 1+eps gets capped, and a negative one below 1-eps. We surface whether this
    // sample's advantage is large enough to be sitting against a clip edge.
    const effective =
      advantage >= 0
        ? Math.min(ratio, clipHigh) * advantage
        : Math.max(ratio, clipLow) * advantage;
    const clipped = Math.abs(advantage) > 1.0; // big advantages ride the clip
    if (clipped) clippedCount += 1;
    for (let c = 0; c < KIND_COUNT; c++) {
      const score = (kind === c ? 1 : 0) - at(oldProbs, c);
      grad[c] = at(grad, c) + effective * score;
    }
    return {
      kindIndex: kind,
      reward: at(rewards, i),
      advantage,
      ratio: clippedRatio,
      clipped,
    };
  });

  // KL term, pulling the policy back toward the frozen reference. We add its
  // negative gradient so the loss minimizes divergence. Schulman's unbiased
  // estimator (equation 4) is what we report as the KL readout.
  const refProbs = softmax(state.refLogits);
  for (let c = 0; c < KIND_COUNT; c++) {
    grad[c] = at(grad, c) - params.klCoef * (at(oldProbs, c) - at(refProbs, c));
  }

  const scale = params.learningRate / groupSize;
  const logits = state.logits.map((value, c) => value + scale * at(grad, c));

  const newProbs = softmax(logits);
  const klToRef = newProbs.reduce((sum, p, c) => {
    const ref = at(refProbs, c);
    const r = ref / Math.max(p, 1e-9);
    return sum + p * (r - Math.log(r) - 1);
  }, 0);

  const next: SimState = {
    ...state,
    logits,
    oldLogits: [...state.logits],
    rngCursor: state.rngCursor + 1,
    step: state.step + 1,
    lastGroup: samples,
    lastStats: {
      groupMean,
      groupStd,
      klToRef,
      clippedFraction: clippedCount / groupSize,
    },
  };

  return {
    ...next,
    history: [...state.history, snapshotPoint(next, klToRef)].slice(-160),
  };
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: GrpoParams;
  startLogits: number[];
}

// A starting policy that already covers all kinds but spreads its mass, so the
// correct kinds are reachable and the GRPO step has room to sharpen onto them.
const SPREAD_START = [0.2, 0.0, 0.3, 0.2, 0.1];

// A starting policy with effectively no mass on either correct kind, so a run of
// normal length never samples a correct answer and GRPO has nothing to reinforce:
// reweighting what the model already produces cannot conjure an answer it does
// not sample. The correct logits sit far below the wrong ones, putting each
// correct kind near a 1-in-10000 chance per draw.
const STARVED_START = [-9, -9, 1.1, 0.9, 0.4];

export const PRESETS: Preset[] = [
  {
    id: "sharpen",
    label: "Sharpen onto correct",
    blurb:
      "The healthy GRPO run. Play it and watch the mass climb onto the correct kinds while the group mean rises and the KL term keeps the policy near the reference.",
    params: {
      groupSize: 16,
      klCoef: 0.04,
      clipEps: 0.2,
      learningRate: 0.6,
      rewardNoise: 0,
      normalizeStd: true,
      hackKind: -1,
      hackBonus: 0,
    },
    startLogits: [...SPREAD_START],
  },
  {
    id: "maj-not-pass",
    label: "Maj@K up, Pass@K flat",
    blurb:
      "The paper's headline finding. Play it and watch Maj@K climb as the policy sharpens onto a correct kind, while Pass@K barely moves because the run never invents a new correct answer.",
    params: {
      groupSize: 16,
      klCoef: 0.04,
      clipEps: 0.2,
      learningRate: 0.7,
      rewardNoise: 0,
      normalizeStd: true,
      hackKind: -1,
      hackBonus: 0,
    },
    startLogits: [...SPREAD_START],
  },
  {
    id: "reward-hack",
    label: "Reward hacking",
    blurb:
      "Point the reward model at a wrong kind and hand it a fat bonus. GRPO trusts the score, so the mass marches onto the wrong answer and the correct kinds drain away.",
    params: {
      groupSize: 16,
      klCoef: 0.02,
      clipEps: 0.2,
      learningRate: 0.7,
      rewardNoise: 0,
      normalizeStd: true,
      hackKind: 2,
      hackBonus: 1.5,
    },
    startLogits: [...SPREAD_START],
  },
  {
    id: "no-headroom",
    label: "No headroom",
    blurb:
      "The policy starts with almost no mass on any correct kind. Play it and watch Maj@K stay stuck: reweighting cannot lift an answer the model never samples.",
    params: {
      groupSize: 16,
      klCoef: 0.04,
      clipEps: 0.2,
      learningRate: 0.7,
      rewardNoise: 0,
      normalizeStd: true,
      hackKind: -1,
      hackBonus: 0,
    },
    startLogits: [...STARVED_START],
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function emptyGroup(): Sample[] {
  return [];
}

function emptyStats(): StepStats {
  return { groupMean: 0, groupStd: 0, klToRef: 0, clippedFraction: 0 };
}

export function initialState(presetId = "sharpen"): SimState {
  const preset = getPreset(presetId);
  const logits = [...preset.startLogits];
  const base: SimState = {
    params: { ...preset.params },
    seed: 1,
    rngCursor: 0,
    logits,
    refLogits: [...logits],
    oldLogits: [...logits],
    step: 0,
    lastGroup: emptyGroup(),
    lastStats: emptyStats(),
    history: [],
    log: [],
  };
  return { ...base, history: [snapshotPoint(base, 0)] };
}

function record(state: SimState, label: string): SimState {
  const entry: LogEntry = {
    id: state.log.length,
    step: state.step,
    label,
    snapshot: state,
  };
  return { ...state, log: [...state.log, entry].slice(-60) };
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  const { params } = state;
  switch (action.type) {
    case "setGroupSize":
      return {
        state: { ...state, params: { ...params, groupSize: action.value } },
        label: `group size -> ${action.value}`,
      };
    case "setKlCoef":
      return {
        state: { ...state, params: { ...params, klCoef: action.value } },
        label: `KL coef -> ${action.value.toFixed(3)}`,
      };
    case "setClipEps":
      return {
        state: { ...state, params: { ...params, clipEps: action.value } },
        label: `clip eps -> ${action.value.toFixed(2)}`,
      };
    case "setLearningRate":
      return {
        state: { ...state, params: { ...params, learningRate: action.value } },
        label: `step size -> ${action.value.toFixed(2)}`,
      };
    case "setRewardNoise":
      return {
        state: { ...state, params: { ...params, rewardNoise: action.value } },
        label: `reward noise -> ${action.value.toFixed(2)}`,
      };
    case "toggleNormalizeStd": {
      const next = !params.normalizeStd;
      return {
        state: { ...state, params: { ...params, normalizeStd: next } },
        label: `divide by std ${next ? "on" : "off"}`,
      };
    }
    case "setHack":
      return {
        state: { ...state, params: { ...params, hackKind: action.kindIndex } },
        label:
          action.kindIndex < 0
            ? "reward hack off"
            : `reward model favors '${ANSWER_KINDS[action.kindIndex]?.label ?? "?"}'`,
      };
    case "setHackBonus":
      return {
        state: { ...state, params: { ...params, hackBonus: action.value } },
        label: `hack bonus -> ${action.value.toFixed(1)}`,
      };
    case "setSeed":
      return {
        state: { ...state, seed: action.value, rngCursor: 0 },
        label: `seed -> ${action.value}`,
      };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: initialState(preset.id),
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return { state: action.snapshot, label: action.label };
  }
}

// The pure reducer. A tick runs one GRPO step; an intervention changes a knob and
// records what happened. Same seed plus same inputs reproduce the run exactly.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const next = grpoStep(state);
    return record(next, `step ${next.step}`);
  }
  const { state: changed, label } = applyIntervention(state, input.action);
  return record(changed, label);
}

export function policyProbs(state: SimState): number[] {
  return softmax(state.logits);
}

export function referenceProbs(state: SimState): number[] {
  return softmax(state.refLogits);
}
