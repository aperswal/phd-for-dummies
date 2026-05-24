// A pure, deterministic model of PRIME (Process Reinforcement through Implicit
// Rewards, arXiv 2502.01456). The paper's one move is that a language model
// trained with cross-entropy on outcome labels already carries a dense per-token
// reward, the log-ratio of its own next-token probability to a frozen
// reference's:
//
//   r_phi(y_t) = beta * log( pi_phi(y_t | y_<t) / pi_ref(y_t | y_<t) )      (Eq 3)
//
// So you get step-level credit assignment for free, with no step labels, and you
// keep that reward model honest by updating it online on fresh rollouts against
// the same outcome labels (the cross-entropy loss in Algorithm 1, line 8). Stop
// updating it and it drifts, the policy games the stale reward, and the reported
// reward climbs while true accuracy stalls (the paper's Figure 5).
//
// This model runs a tiny training loop the reader can perturb. The "policy" is a
// distribution over short reasoning traces on one toy task. Each trace is a path
// through a few reasoning "moves"; one path is genuinely correct, and one is a
// shortcut that the verifier marks wrong but that a drifting reward model can be
// fooled into liking. Each training step samples K traces, grades them with an
// exact outcome verifier, scores each token with the implicit reward, builds the
// RLOO-with-implicit-process-rewards advantage of Eq 5, updates the policy, and
// (when online) updates the implicit PRM by cross-entropy on the outcome labels.
//
// Everything is exact and seeded. No Math.random, no wall clock. Same seed plus
// the same inputs reproduce a run token for token, which is what lets the event
// log rewind.

// ---------------------------------------------------------------------------
// Self-contained primitives (kept local on purpose so this simulation owns its
// engine and does not depend on shared sim modules other work may be editing).
// ---------------------------------------------------------------------------

// One step of the mulberry32 recurrence as a pure function: given the state,
// return the next value in [0, 1) and the next state, so a reducer can thread a
// seed through immutable state and stay deterministic.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

export interface SimEvent<TSnapshot> {
  id: number;
  tick: number;
  label: string;
  snapshot: TSnapshot;
}

export interface EventLog<TSnapshot> {
  events: SimEvent<TSnapshot>[];
  nextEventId: number;
}

export function emptyLog<TSnapshot>(): EventLog<TSnapshot> {
  return { events: [], nextEventId: 1 };
}

const LOG_LIMIT = 60;

function record<TSnapshot>(
  log: EventLog<TSnapshot>,
  tick: number,
  label: string,
  snapshot: TSnapshot,
): EventLog<TSnapshot> {
  const event: SimEvent<TSnapshot> = {
    id: log.nextEventId,
    tick,
    label,
    snapshot,
  };
  return {
    events: [...log.events, event].slice(-LOG_LIMIT),
    nextEventId: log.nextEventId + 1,
  };
}

// ---------------------------------------------------------------------------
// The toy task: three reasoning moves the policy can take at the single branch.
// "correct" is the real solution path. "shortcut" reaches the right number by a
// wrong process the verifier rejects but a drifting reward model can be tricked
// into rewarding. "wrong" is plainly bad.
// ---------------------------------------------------------------------------

export type Move = "correct" | "shortcut" | "wrong";
export const MOVES: Move[] = ["correct", "shortcut", "wrong"];

export const MOVE_LABELS: Record<Move, string> = {
  correct: "sound step",
  shortcut: "lucky shortcut",
  wrong: "wrong step",
};

// The outcome verifier (rule-based exact match, the paper's r_o). It rewards
// only the genuinely sound path. The shortcut lands the right final number but
// through a process the verifier rejects, so it scores 0. This is the gap
// between "right answer" and "right reasoning" the paper cares about.
export function outcomeReward(move: Move): number {
  return move === "correct" ? 1 : 0;
}

// Per-token implicit reward magnitude built into each move's tokens. A trace is
// a short sequence of tokens; the reference model pi_ref assigns each a baseline
// log-prob and the implicit PRM pi_phi assigns its own. The model carries the
// PRM's accumulated log-ratio per move as a single scalar "preference" so the
// per-token reward of a move is beta times that preference (Eq 3, summed over
// the trace's tokens).
export const TRACE_TOKENS: Record<Move, number> = {
  correct: 4,
  shortcut: 3,
  wrong: 4,
};

// ---------------------------------------------------------------------------
// Parameters and state.
// ---------------------------------------------------------------------------

export interface PrimeParams {
  beta: number; // implicit-reward temperature in Eq 3
  rollouts: number; // K responses per prompt (leave-one-out baseline size)
  prmOnline: boolean; // update the implicit PRM on rollouts (Algorithm 1 line 8)
  prmFromScratch: boolean; // PRM init: false = SFT model, true = zero reward
  policyLr: number; // policy gradient step size
  prmLr: number; // implicit-PRM cross-entropy step size
  seed: number;
}

export interface PrimeState {
  params: PrimeParams;
  tick: number;
  // policyLogits[move] are the policy's unnormalized scores over the branch.
  policyLogits: Record<Move, number>;
  // prmPref[move] is the implicit PRM's accumulated log(pi_phi / pi_ref) per
  // token for that move's trace, the thing beta multiplies into the reward.
  prmPref: Record<Move, number>;
  // Read-outs the view shows. trueAccuracy is the fraction of probability mass
  // on the genuinely correct move. reportedReward is the mean process+outcome
  // reward the training loop sees. prmAccuracy is how well the PRM ranks correct
  // over shortcut (the paper's Figure 5 metric).
  trueAccuracy: number;
  reportedReward: number;
  prmAccuracy: number;
  hackGap: number; // reportedReward (normalized) minus trueAccuracy
  lastBatch: BatchRow[];
  // A bounded trace of the three read-outs per training step, so the view can
  // draw the curves the paper plots in Figures 3 and 5 without re-simulating.
  history: HistoryPoint[];
  rng: number;
  log: EventLog<PrimeSnapshot>;
}

export interface HistoryPoint {
  tick: number;
  trueAccuracy: number;
  reportedReward: number;
  prmAccuracy: number;
}

const HISTORY_LIMIT = 200;

export interface BatchRow {
  move: Move;
  outcome: number;
  processReward: number; // beta * sum of per-token implicit rewards over trace
  advantage: number; // Eq 5: RLOO(process) + RLOO(outcome)
}

// A snapshot is the full mutable state minus the log, so the event log can
// rewind a run to exactly the moment before an intervention.
export interface PrimeSnapshot {
  params: PrimeParams;
  tick: number;
  policyLogits: Record<Move, number>;
  prmPref: Record<Move, number>;
  rng: number;
}

function softmax3(logits: Record<Move, number>): Record<Move, number> {
  const values = MOVES.map((m) => logits[m]);
  const max = Math.max(...values);
  const exps = MOVES.map((m) => Math.exp(logits[m] - max));
  const total = exps.reduce((s, v) => s + v, 0) || 1;
  const out = {} as Record<Move, number>;
  MOVES.forEach((m, i) => {
    out[m] = at(exps, i) / total;
  });
  return out;
}

// ---------------------------------------------------------------------------
// The implicit PRM's per-token log-ratio log(pi_phi / pi_ref) at init, taken
// from the SFT model. It reads the surface form, and the shortcut reads fluently
// even though its process is wrong, so the SFT-initialized PRM over-likes the
// shortcut. That is exactly why a static reward model is dangerous: the surface
// form looks good, so a frozen reward keeps paying out for it even as the policy
// drifts toward gaming it. The online cross-entropy update is what fixes this.
// ---------------------------------------------------------------------------

const SFT_INIT_PREF: Record<Move, number> = {
  correct: 0.2,
  shortcut: 0.9,
  wrong: -0.6,
};

function defaultParams(): PrimeParams {
  return {
    beta: 0.6,
    rollouts: 4,
    prmOnline: true,
    prmFromScratch: false,
    policyLr: 0.6,
    prmLr: 0.5,
    seed: 7,
  };
}

function initialPrmPref(params: PrimeParams): Record<Move, number> {
  // SFT-initialized PRM starts from the SFT model's surface preferences, which
  // over-like the fluent shortcut. A from-scratch PRM starts flat at zero on
  // every move, so it hands out zero process reward until the online update
  // teaches it (the paper's "trained from scratch" run that still works because
  // of the online update).
  if (params.prmFromScratch) {
    return { correct: 0, shortcut: 0, wrong: 0 };
  }
  return { ...SFT_INIT_PREF };
}

// ---------------------------------------------------------------------------
// Derived read-outs.
// ---------------------------------------------------------------------------

function processReward(
  params: PrimeParams,
  state: ProcessInputs,
  move: Move,
): number {
  // Eq 3 summed over the trace: beta * log(pi_phi / pi_ref) per token. We carry
  // that accumulated log-ratio per move as one scalar prmPref, so the trace
  // reward is beta * prmPref * tokenCount, scaled down to keep it on the same
  // order as the 0/1 outcome reward.
  const perToken = state.prmPref[move];
  return params.beta * perToken * TRACE_TOKENS[move] * 0.5;
}

interface ProcessInputs {
  prmPref: Record<Move, number>;
}

function prmRanksCorrect(state: ProcessInputs): number {
  // PRM accuracy: does the implicit reward rank the sound step above the
  // shortcut? A soft margin so it reads as a probability, matching Figure 5's
  // "PRM classification accuracy".
  const margin = state.prmPref.correct - state.prmPref.shortcut;
  return 1 / (1 + Math.exp(-4 * margin));
}

function computeReadouts(
  params: PrimeParams,
  policyLogits: Record<Move, number>,
  prmPref: Record<Move, number>,
): {
  trueAccuracy: number;
  reportedReward: number;
  prmAccuracy: number;
  hackGap: number;
} {
  const policy = softmax3(policyLogits);
  const trueAccuracy = policy.correct;

  const inputs: ProcessInputs = { prmPref };
  let reported = 0;
  for (const move of MOVES) {
    const reward = outcomeReward(move) + processReward(params, inputs, move);
    reported += policy[move] * reward;
  }
  // Normalize reported reward into roughly [0, 1] so the gap against true
  // accuracy is legible. The constant just rescales; it changes no dynamics.
  const reportedNorm = Math.max(0, Math.min(1, reported / 1.6 + 0.1));

  return {
    trueAccuracy,
    reportedReward: reportedNorm,
    prmAccuracy: prmRanksCorrect(inputs),
    hackGap: reportedNorm - trueAccuracy,
  };
}

function withReadouts(state: PrimeState): PrimeState {
  const r = computeReadouts(state.params, state.policyLogits, state.prmPref);
  return { ...state, ...r };
}

function pushHistory(state: PrimeState): PrimeState {
  const point: HistoryPoint = {
    tick: state.tick,
    trueAccuracy: state.trueAccuracy,
    reportedReward: state.reportedReward,
    prmAccuracy: state.prmAccuracy,
  };
  return { ...state, history: [...state.history, point].slice(-HISTORY_LIMIT) };
}

export function initialState(presetId = "online"): PrimeState {
  const params = { ...getPreset(presetId).params };
  const base: PrimeState = {
    params,
    tick: 0,
    policyLogits: { correct: 0, shortcut: 0, wrong: 0 },
    prmPref: initialPrmPref(params),
    trueAccuracy: 0,
    reportedReward: 0,
    prmAccuracy: 0,
    hackGap: 0,
    lastBatch: [],
    history: [],
    rng: params.seed >>> 0,
    log: emptyLog(),
  };
  return pushHistory(withReadouts(base));
}

// ---------------------------------------------------------------------------
// One training step. This is the whole of Algorithm 1's loop body on the toy
// task: sample K rollouts, grade them, score tokens with the implicit reward,
// build the Eq 5 advantage, update the policy, then (online) update the PRM.
// ---------------------------------------------------------------------------

function sampleMove(policy: Record<Move, number>, draw: number): Move {
  let acc = 0;
  for (const move of MOVES) {
    acc += policy[move];
    if (draw <= acc) return move;
  }
  return "wrong";
}

function trainingStep(state: PrimeState): PrimeState {
  const { params } = state;
  const policy = softmax3(state.policyLogits);
  const inputs: ProcessInputs = { prmPref: state.prmPref };

  // 1-2. Sample K rollouts and grade each with the outcome verifier and the
  // implicit process reward.
  let rng = state.rng;
  const moves: Move[] = [];
  const outcomes: number[] = [];
  const processes: number[] = [];
  for (let k = 0; k < params.rollouts; k++) {
    const draw = nextRandom(rng);
    rng = draw.state;
    const move = sampleMove(policy, draw.value);
    moves.push(move);
    outcomes.push(outcomeReward(move));
    processes.push(processReward(params, inputs, move));
  }

  // 3. Eq 5 advantage: a leave-one-out baseline on the process rewards plus a
  // leave-one-out baseline on the outcome rewards, summed. With K rollouts the
  // baseline for sample i is the mean of the other K-1.
  const advantages = moves.map((_, i) => {
    const looProcess = leaveOneOut(processes, i);
    const looOutcome = leaveOneOut(outcomes, i);
    return looProcess + looOutcome;
  });

  // 4. Policy gradient step. Push the policy logits toward moves with positive
  // advantage and away from negative ones, averaged over the batch.
  const policyLogits = { ...state.policyLogits };
  moves.forEach((move, i) => {
    policyLogits[move] +=
      (params.policyLr * at(advantages, i)) / params.rollouts;
  });

  // 5. Online implicit-PRM update by cross-entropy on the outcome labels
  // (Algorithm 1 line 8). The CE loss pushes the PRM's preference up on rollouts
  // the verifier marked correct and down on the rest, recomputed only on this
  // batch of fresh policy rollouts. That is what keeps it calibrated under the
  // shifting policy. Offline, the PRM never moves and drifts out of date.
  const prmPref = { ...state.prmPref };
  if (params.prmOnline) {
    moves.forEach((move, i) => {
      const target = at(outcomes, i); // 1 if verifier liked it, else 0
      const predicted = 1 / (1 + Math.exp(-prmPref[move]));
      prmPref[move] += params.prmLr * (target - predicted);
    });
  }

  const lastBatch: BatchRow[] = moves.map((move, i) => ({
    move,
    outcome: at(outcomes, i),
    processReward: at(processes, i),
    advantage: at(advantages, i),
  }));

  const advanced: PrimeState = {
    ...state,
    tick: state.tick + 1,
    policyLogits,
    prmPref,
    lastBatch,
    rng,
  };
  return withReadouts(advanced);
}

function leaveOneOut(values: number[], index: number): number {
  if (values.length <= 1) return at(values, index);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    if (i !== index) sum += at(values, i);
  }
  const baseline = sum / (values.length - 1);
  return at(values, index) - baseline;
}

// ---------------------------------------------------------------------------
// Interventions: everything the reader can do mid-run.
// ---------------------------------------------------------------------------

export type Intervention =
  | { type: "togglePrmOnline" }
  | { type: "togglePrmFromScratch" }
  | { type: "setBeta"; value: number }
  | { type: "setRollouts"; value: number }
  | { type: "setPolicyLr"; value: number }
  | { type: "setPrmLr"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "injectShortcut" } // make the shortcut more tempting to the policy
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: PrimeSnapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function snapshotOf(state: PrimeState): PrimeSnapshot {
  return {
    params: { ...state.params },
    tick: state.tick,
    policyLogits: { ...state.policyLogits },
    prmPref: { ...state.prmPref },
    rng: state.rng,
  };
}

function fromSnapshot(
  snapshot: PrimeSnapshot,
  log: EventLog<PrimeSnapshot>,
  history: HistoryPoint[],
): PrimeState {
  const base: PrimeState = {
    params: { ...snapshot.params },
    tick: snapshot.tick,
    policyLogits: { ...snapshot.policyLogits },
    prmPref: { ...snapshot.prmPref },
    trueAccuracy: 0,
    reportedReward: 0,
    prmAccuracy: 0,
    hackGap: 0,
    lastBatch: [],
    history: history.filter((point) => point.tick <= snapshot.tick),
    rng: snapshot.rng,
    log,
  };
  return withReadouts(base);
}

function applyIntervention(
  state: PrimeState,
  action: Intervention,
): { state: PrimeState; label: string } {
  switch (action.type) {
    case "togglePrmOnline": {
      const prmOnline = !state.params.prmOnline;
      return {
        state: { ...state, params: { ...state.params, prmOnline } },
        label: `PRM update ${prmOnline ? "online" : "frozen"}`,
      };
    }
    case "togglePrmFromScratch": {
      const prmFromScratch = !state.params.prmFromScratch;
      const params = { ...state.params, prmFromScratch };
      return {
        state: withReadouts({
          ...state,
          params,
          prmPref: initialPrmPref(params),
        }),
        label: `PRM init ${prmFromScratch ? "from scratch" : "from SFT"}`,
      };
    }
    case "setBeta":
      return {
        state: withReadouts({
          ...state,
          params: { ...state.params, beta: action.value },
        }),
        label: `beta -> ${action.value.toFixed(2)}`,
      };
    case "setRollouts":
      return {
        state: {
          ...state,
          params: { ...state.params, rollouts: action.value },
        },
        label: `K -> ${action.value}`,
      };
    case "setPolicyLr":
      return {
        state: {
          ...state,
          params: { ...state.params, policyLr: action.value },
        },
        label: `policy lr -> ${action.value.toFixed(2)}`,
      };
    case "setPrmLr":
      return {
        state: { ...state, params: { ...state.params, prmLr: action.value } },
        label: `PRM lr -> ${action.value.toFixed(2)}`,
      };
    case "setSeed":
      return {
        state: { ...state, params: { ...state.params, seed: action.value } },
        label: `seed -> ${action.value}`,
      };
    case "injectShortcut": {
      // Make the shortcut a stronger attractor for the policy by nudging its
      // logit up. A frozen reward model never notices; an online one re-grades
      // it as wrong and pulls the PRM reward back down.
      const policyLogits = {
        ...state.policyLogits,
        shortcut: state.policyLogits.shortcut + 1.5,
      };
      return {
        state: withReadouts({ ...state, policyLogits }),
        label: "inject shortcut temptation",
      };
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: initialState(action.id),
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return {
        state: fromSnapshot(action.snapshot, state.log, state.history),
        label: action.label,
      };
  }
}

// The pure reducer. A tick runs one training step; an intervention changes the
// run and records a snapshot so the reader can rewind. Every step is
// deterministic given the seed.
export function step(state: PrimeState, input: Input): PrimeState {
  if (input.kind === "tick") {
    const advanced = pushHistory(trainingStep(state));
    return {
      ...advanced,
      log: record(
        state.log,
        advanced.tick,
        `step ${advanced.tick}`,
        snapshotOf(state),
      ),
    };
  }

  const { state: next, label } = applyIntervention(state, input.action);
  return {
    ...next,
    log: record(state.log, state.tick, label, snapshotOf(state)),
  };
}

// ---------------------------------------------------------------------------
// Presets: a happy path and the paper's failure case.
// ---------------------------------------------------------------------------

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: PrimeParams;
}

export const PRESETS: Preset[] = [
  {
    id: "online",
    label: "Online PRIME",
    blurb:
      "The full method. The implicit PRM starts from the SFT model and updates on every batch of rollouts. Play and watch true accuracy and reported reward climb together while PRM accuracy holds.",
    params: { ...defaultParams() },
  },
  {
    id: "frozen",
    label: "Frozen reward hacks",
    blurb:
      "The implicit PRM is frozen after init, like a static reward model, and beta is turned up so its stale process reward carries weight. Play, then hit Inject shortcut: the reported reward keeps climbing while true accuracy falls and PRM accuracy slides. That gap is reward hacking.",
    params: { ...defaultParams(), prmOnline: false, beta: 1.2 },
  },
  {
    id: "scratch",
    label: "PRM from scratch",
    blurb:
      "The implicit PRM starts at zero reward on every move, so early steps lean only on the outcome verifier. The online cross-entropy update teaches the PRM to rank the sound step over the shortcut as training goes.",
    params: { ...defaultParams(), prmFromScratch: true },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

// A convenience for the view and tests.
export function policyDistribution(state: PrimeState): Record<Move, number> {
  return softmax3(state.policyLogits);
}

export { snapshotOf };
