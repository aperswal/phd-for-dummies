// A pure, deterministic model of turn-level credit assignment from "Reinforcing
// Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design". The setting
// is the paper's two-turn (K = 2) search agent: turn 1 reasons and calls a
// search tool, turn 2 reasons over the result and answers. Each rollout in a
// group earns a turn-1 intermediate reward R^I (tool execution + retrieval +
// format - search penalty) and a turn-2 outcome reward R^O (exact match + format).
//
// The model computes the advantage each algorithm assigns to each turn, then
// steps a tiny training loop. The policy is a set of per-trait propensities; on
// each training step the group's advantages push those propensities up or down,
// which changes the sampled rollouts on the next step. GRPO-OR (outcome only,
// trajectory-level) starves turn 1 of any gradient, so the search behavior
// drifts and collapses exactly as Figure 1 shows. MT-GRPO assigns a turn-1
// advantage of A^I + alpha*A^O, so good searches keep getting reinforced and
// tool use stays stable. No Math.random anywhere; a seeded mulberry32 stream
// makes a given seed reproduce the run exactly.

// One step of the mulberry32 recurrence as a pure function. Threading the
// generator state through immutable sim state keeps every step deterministic and
// replayable instead of relying on a stateful closure.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

// Indexed read that throws on a genuine out-of-range access instead of quietly
// returning undefined, so a real bug fails loudly while the hot loops stay typed.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

export type AlgorithmId = "grpo-or" | "grpo-mr" | "mt-grpo";

export interface Algorithm {
  id: AlgorithmId;
  name: string;
  blurb: string;
  granularity: string;
}

export const ALGORITHMS: Algorithm[] = [
  {
    id: "grpo-or",
    name: "GRPO-OR",
    blurb:
      "Outcome reward only, one advantage for the whole trajectory. Turn 1 gets no signal of its own, so the search behavior has no reason to stay.",
    granularity: "trajectory-level",
  },
  {
    id: "grpo-mr",
    name: "GRPO-MR",
    blurb:
      "Merges the turn-1 and turn-2 rewards into one number, then normalizes once across the group. Better than outcome-only, but still one advantage spread over every token.",
    granularity: "trajectory-level",
  },
  {
    id: "mt-grpo",
    name: "MT-GRPO",
    blurb:
      "Normalizes turn 1 and turn 2 separately, then gives turn 1 its own advantage plus a discounted share of the outcome. Credit lands on the turn that earned it.",
    granularity: "turn-level",
  },
];

export function getAlgorithm(id: AlgorithmId): Algorithm {
  return (
    ALGORITHMS.find((algorithm) => algorithm.id === id) ?? at(ALGORITHMS, 0)
  );
}

// The policy is four propensities in [0, 1], one per behavior the agent can
// learn. They decide, per rollout, how likely the agent is to do the right thing
// at turn 1 (call the tool well, retrieve the answer, format the call) and at
// turn 2 (give a correct, well-formatted answer). The training loop nudges these.
export interface Policy {
  toolUse: number; // P(turn-1 tool call is well formed and executes)
  retrieval: number; // P(the search actually returns the ground-truth answer)
  intermediateFormat: number; // P(turn-1 XML format is correct)
  answer: number; // P(turn-2 answer is exact-match correct)
}

// A single rollout in a group: which behaviors fired, and the resulting per-turn
// rewards under the paper's case-study reward design (Section 5.2 / Appendix E.2).
export interface Rollout {
  id: number;
  toolExecuted: boolean;
  retrieved: boolean;
  intermediateFormatOk: boolean;
  searchCount: number;
  answerCorrect: boolean;
  outcomeFormatOk: boolean;
  intermediateReward: number; // R^I
  outcomeReward: number; // R^O
  mergedReward: number; // R^I + gamma * R^O, the GRPO-MR signal
}

export interface RewardWeights {
  toolExecution: number; // 0.2 when the tool runs (Appendix E.2)
  retrieval: number; // 0.3 / 0.5 when the answer is in the result
  intermediateFormat: number; // +0.1 ok, -0.2 broken
  searchPenalty: number; // lambda_s per search, default 0.1
  exactMatch: number; // outcome: 1.0 correct, 0.2 formatted-but-wrong, -1 broken
  alpha: number; // discount on the outcome term inside the turn-1 advantage
}

export const DEFAULT_WEIGHTS: RewardWeights = {
  toolExecution: 0.2,
  retrieval: 0.5,
  intermediateFormat: 0.1,
  searchPenalty: 0.1,
  exactMatch: 1.0,
  alpha: 1.0,
};

function bernoulli(probability: number, draw: number): boolean {
  return draw < probability;
}

function intermediateReward(rollout: Rollout, weights: RewardWeights): number {
  const tool = rollout.toolExecuted ? weights.toolExecution : 0;
  const retrieval = rollout.retrieved ? weights.retrieval : 0;
  const format = rollout.intermediateFormatOk
    ? weights.intermediateFormat
    : -0.2;
  const penalty = weights.searchPenalty * rollout.searchCount;
  return tool + retrieval + format - penalty;
}

function outcomeReward(rollout: Rollout, weights: RewardWeights): number {
  if (!rollout.outcomeFormatOk) return -1;
  if (rollout.answerCorrect) return weights.exactMatch;
  return 0.2;
}

// Draws one group of G rollouts from the current policy using the seeded stream.
// Returns the rollouts plus the advanced rng state so the caller stays pure.
function sampleGroup(
  policy: Policy,
  weights: RewardWeights,
  groupSize: number,
  rngState: number,
): { rollouts: Rollout[]; rngState: number } {
  const rollouts: Rollout[] = [];
  let state = rngState;
  for (let i = 0; i < groupSize; i++) {
    const draws: number[] = [];
    for (let d = 0; d < 5; d++) {
      const next = nextRandom(state);
      state = next.state;
      draws.push(next.value);
    }
    const toolExecuted = bernoulli(policy.toolUse, at(draws, 0));
    // Retrieval can only succeed if the tool actually ran.
    const retrieved = toolExecuted && bernoulli(policy.retrieval, at(draws, 1));
    const intermediateFormatOk = bernoulli(
      policy.intermediateFormat,
      at(draws, 2),
    );
    // A weak agent over-searches; a confident one searches once. Tie the count
    // to how unsure the policy is so the search penalty has something to bite.
    const extraSearch = !toolExecuted || at(draws, 3) > policy.retrieval;
    const searchCount = toolExecuted ? (extraSearch ? 2 : 1) : 0;
    const answerCorrect = bernoulli(policy.answer, at(draws, 4));
    const outcomeFormatOk = intermediateFormatOk || at(draws, 2) < 0.85;

    const partial: Rollout = {
      id: i,
      toolExecuted,
      retrieved,
      intermediateFormatOk,
      searchCount,
      answerCorrect,
      outcomeFormatOk,
      intermediateReward: 0,
      outcomeReward: 0,
      mergedReward: 0,
    };
    partial.intermediateReward = intermediateReward(partial, weights);
    partial.outcomeReward = outcomeReward(partial, weights);
    partial.mergedReward = partial.intermediateReward + partial.outcomeReward;
    rollouts.push(partial);
  }
  return { rollouts, rngState: state };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length === 0) return 0;
  const m = mean(values);
  const variance = mean(values.map((value) => (value - m) * (value - m)));
  return Math.sqrt(variance);
}

// Group-relative normalization, exactly the GRPO advantage definition:
// A_i = (R_i - mean(R)) / std(R). A flat group (everyone equal) gives zero
// advantage, which is the degenerate case GRPO-OR keeps falling into.
function normalize(values: number[]): number[] {
  const m = mean(values);
  const s = std(values);
  if (s < 1e-6) return values.map(() => 0);
  return values.map((value) => (value - m) / s);
}

export interface AdvantageRow {
  rolloutId: number;
  turn1: number; // advantage applied to every turn-1 token
  turn2: number; // advantage applied to every turn-2 token
}

// The heart of the paper. Given a group of rollouts, compute the per-turn
// advantage each algorithm assigns. This is equations 4, 5, 6, 10, 11, 12.
export function computeAdvantages(
  rollouts: Rollout[],
  algorithm: AlgorithmId,
  weights: RewardWeights,
): AdvantageRow[] {
  const intermediate = rollouts.map((r) => r.intermediateReward);
  const outcome = rollouts.map((r) => r.outcomeReward);
  const merged = rollouts.map((r) => r.mergedReward);

  if (algorithm === "grpo-or") {
    // One advantage from outcome reward only, applied to the whole trajectory.
    const a = normalize(outcome);
    return rollouts.map((r, i) => ({
      rolloutId: r.id,
      turn1: at(a, i),
      turn2: at(a, i),
    }));
  }

  if (algorithm === "grpo-mr") {
    // One advantage from the merged reward, applied to the whole trajectory.
    const a = normalize(merged);
    return rollouts.map((r, i) => ({
      rolloutId: r.id,
      turn1: at(a, i),
      turn2: at(a, i),
    }));
  }

  // MT-GRPO: normalize the two turns separately, then turn 1 gets its own
  // intermediate advantage plus a discounted share of the outcome advantage.
  const aIntermediate = normalize(intermediate);
  const aOutcome = normalize(outcome);
  return rollouts.map((r, i) => ({
    rolloutId: r.id,
    turn1: at(aIntermediate, i) + weights.alpha * at(aOutcome, i),
    turn2: at(aOutcome, i),
  }));
}

const LEARNING_RATE = 0.06;
const DRIFT = 0.012; // unsupervised behaviors slowly forget, the source of collapse

function clamp01(value: number): number {
  return Math.min(1, Math.max(0.02, value));
}

// One policy-gradient update. For each behavior, we correlate whether it fired
// in a rollout with the advantage that rollout's relevant turn received, then
// nudge the propensity toward behaviors the advantage rewarded. A behavior that
// no advantage ever touches drifts down on its own (the forgetting that sinks
// GRPO-OR's tool use).
function updatePolicy(
  policy: Policy,
  rollouts: Rollout[],
  advantages: AdvantageRow[],
): Policy {
  const advById = new Map(advantages.map((a) => [a.rolloutId, a]));
  function gradient(
    fired: (r: Rollout) => boolean,
    turn: (a: AdvantageRow) => number,
  ): number {
    let signal = 0;
    for (const rollout of rollouts) {
      const adv = advById.get(rollout.id);
      if (!adv) continue;
      const sign = fired(rollout) ? 1 : -1;
      signal += sign * turn(adv);
    }
    return signal / Math.max(rollouts.length, 1);
  }

  const toolGrad = gradient(
    (r) => r.toolExecuted,
    (a) => a.turn1,
  );
  const retrievalGrad = gradient(
    (r) => r.retrieved,
    (a) => a.turn1,
  );
  const formatGrad = gradient(
    (r) => r.intermediateFormatOk,
    (a) => a.turn1,
  );
  const answerGrad = gradient(
    (r) => r.answerCorrect,
    (a) => a.turn2,
  );

  return {
    toolUse: clamp01(policy.toolUse + LEARNING_RATE * toolGrad - DRIFT),
    retrieval: clamp01(
      policy.retrieval + LEARNING_RATE * retrievalGrad - DRIFT,
    ),
    intermediateFormat: clamp01(
      policy.intermediateFormat + LEARNING_RATE * formatGrad - DRIFT,
    ),
    answer: clamp01(policy.answer + LEARNING_RATE * answerGrad),
  };
}

export interface MetricPoint {
  step: number;
  toolUse: number;
  retrieval: number;
  answer: number;
}

export interface SimState {
  algorithm: AlgorithmId;
  weights: RewardWeights;
  groupSize: number;
  seed: number;
  step: number;
  rngState: number;
  policy: Policy;
  rollouts: Rollout[];
  advantages: AdvantageRow[];
  history: MetricPoint[];
  log: LogEntry[];
}

export interface LogEntry {
  id: number;
  step: number;
  label: string;
  snapshot: Snapshot;
}

// A snapshot carries everything needed to rewind the run to a moment: the seed
// stream cursor, the policy, and the configuration.
export interface Snapshot {
  algorithm: AlgorithmId;
  weights: RewardWeights;
  groupSize: number;
  seed: number;
  step: number;
  rngState: number;
  policy: Policy;
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  algorithm: AlgorithmId;
  weights: RewardWeights;
  policy: Policy;
  groupSize: number;
  seed: number;
}

const START_POLICY: Policy = {
  toolUse: 0.55,
  retrieval: 0.4,
  intermediateFormat: 0.5,
  answer: 0.3,
};

export const PRESETS: Preset[] = [
  {
    id: "mt-stable",
    label: "MT-GRPO: stable",
    blurb:
      "Turn-level credit. Run it and watch tool use climb and hold while answer accuracy keeps rising. Then switch the algorithm to GRPO-OR mid-run and watch tool use start to slide.",
    algorithm: "mt-grpo",
    weights: DEFAULT_WEIGHTS,
    policy: { ...START_POLICY },
    groupSize: 8,
    seed: 7,
  },
  {
    id: "or-collapse",
    label: "GRPO-OR: collapse",
    blurb:
      "Outcome reward only. Turn 1 gets no signal of its own, so the search behavior forgets itself and the tool-use curve falls, the failure case from Figure 1.",
    algorithm: "grpo-or",
    weights: DEFAULT_WEIGHTS,
    policy: { ...START_POLICY },
    groupSize: 8,
    seed: 7,
  },
  {
    id: "no-penalty",
    label: "Drop the search penalty",
    blurb:
      "MT-GRPO with the search-count penalty set to zero. Nothing discourages over-searching, so the agent leans on the tool more than it needs to.",
    algorithm: "mt-grpo",
    weights: { ...DEFAULT_WEIGHTS, searchPenalty: 0 },
    policy: { ...START_POLICY },
    groupSize: 8,
    seed: 7,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function metricPoint(step: number, policy: Policy): MetricPoint {
  return {
    step,
    toolUse: policy.toolUse,
    retrieval: policy.retrieval,
    answer: policy.answer,
  };
}

function snapshotOf(state: SimState): Snapshot {
  return {
    algorithm: state.algorithm,
    weights: state.weights,
    groupSize: state.groupSize,
    seed: state.seed,
    step: state.step,
    rngState: state.rngState,
    policy: state.policy,
  };
}

export function initialFromPreset(presetId: string): SimState {
  const preset = getPreset(presetId);
  const policy = { ...preset.policy };
  const sampled = sampleGroup(
    policy,
    preset.weights,
    preset.groupSize,
    preset.seed,
  );
  const advantages = computeAdvantages(
    sampled.rollouts,
    preset.algorithm,
    preset.weights,
  );
  return {
    algorithm: preset.algorithm,
    weights: { ...preset.weights },
    groupSize: preset.groupSize,
    seed: preset.seed,
    step: 0,
    rngState: sampled.rngState,
    policy,
    rollouts: sampled.rollouts,
    advantages,
    history: [metricPoint(0, policy)],
    log: [],
  };
}

const MAX_HISTORY = 240;
const MAX_LOG = 60;

function record(state: SimState, label: string): LogEntry[] {
  const entry: LogEntry = {
    id: state.log.length + 1,
    step: state.step,
    label,
    snapshot: snapshotOf(state),
  };
  return [...state.log, entry].slice(-MAX_LOG);
}

export type Intervention =
  | { type: "setAlgorithm"; id: AlgorithmId }
  | { type: "setWeight"; key: keyof RewardWeights; value: number }
  | { type: "setGroupSize"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// Resamples the current group and recomputes advantages under whatever the
// configuration now is, without advancing the training step. Used after any
// intervention so the displayed group and advantages always match the controls.
function resampleCurrent(state: SimState): SimState {
  const sampled = sampleGroup(
    state.policy,
    state.weights,
    state.groupSize,
    state.rngState,
  );
  const advantages = computeAdvantages(
    sampled.rollouts,
    state.algorithm,
    state.weights,
  );
  return {
    ...state,
    rngState: sampled.rngState,
    rollouts: sampled.rollouts,
    advantages,
  };
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setAlgorithm": {
      const next = resampleCurrent({ ...state, algorithm: action.id });
      return {
        ...next,
        log: record(next, `algorithm -> ${getAlgorithm(action.id).name}`),
      };
    }
    case "setWeight": {
      const weights = { ...state.weights, [action.key]: action.value };
      const next = resampleCurrent({ ...state, weights });
      return {
        ...next,
        log: record(next, `${action.key} -> ${action.value.toFixed(2)}`),
      };
    }
    case "setGroupSize": {
      const next = resampleCurrent({ ...state, groupSize: action.value });
      return { ...next, log: record(next, `group size -> ${action.value}`) };
    }
    case "setSeed": {
      const reseeded = resampleCurrent({
        ...state,
        seed: action.value,
        rngState: action.value >>> 0,
      });
      return { ...reseeded, log: record(reseeded, `seed -> ${action.value}`) };
    }
    case "loadPreset": {
      const fresh = initialFromPreset(action.id);
      return {
        ...fresh,
        log: record(fresh, `preset -> ${getPreset(action.id).label}`),
      };
    }
    case "restore": {
      const s = action.snapshot;
      const restored: SimState = {
        ...state,
        algorithm: s.algorithm,
        weights: { ...s.weights },
        groupSize: s.groupSize,
        seed: s.seed,
        step: s.step,
        rngState: s.rngState,
        policy: { ...s.policy },
        history: state.history.filter((point) => point.step <= s.step),
      };
      const resampled = resampleCurrent(restored);
      return { ...resampled, log: record(resampled, action.label) };
    }
  }
}

// The pure reducer. A tick runs one training step: take the current group's
// advantages, update the policy, advance the step, then sample the next group
// the updated policy will train on. Every step is deterministic given the seed.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") {
    return applyIntervention(state, input.action);
  }

  const updated = updatePolicy(state.policy, state.rollouts, state.advantages);
  const nextStep = state.step + 1;
  const sampled = sampleGroup(
    updated,
    state.weights,
    state.groupSize,
    state.rngState,
  );
  const advantages = computeAdvantages(
    sampled.rollouts,
    state.algorithm,
    state.weights,
  );
  const history = [...state.history, metricPoint(nextStep, updated)].slice(
    -MAX_HISTORY,
  );
  return {
    ...state,
    step: nextStep,
    policy: updated,
    rngState: sampled.rngState,
    rollouts: sampled.rollouts,
    advantages,
    history,
  };
}

// Spread of the turn-1 advantages across the group. When this is near zero,
// every rollout got the same credit and turn 1 learned nothing this step, which
// is the dead signal GRPO-OR keeps producing.
export function turn1Spread(advantages: AdvantageRow[]): number {
  return std(advantages.map((a) => a.turn1));
}

// True when turn 1 is receiving a usable gradient. This is the invariant the
// reader watches hold under MT-GRPO and break under GRPO-OR.
export function turn1HasSignal(advantages: AdvantageRow[]): boolean {
  return turn1Spread(advantages) > 0.05;
}
