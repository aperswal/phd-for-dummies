// A pure, deterministic model of Expert Iteration (ExIt) from "Thinking Fast and
// Slow with Deep Learning and Tree Search". ExIt runs a closed loop between two
// players. A slow expert (a tree search) plans strong moves at single positions.
// A fast apprentice (a neural net) imitates the expert across the whole state
// space, then feeds its learned policy back as a prior that biases the next
// search. Each iteration the apprentice chases the expert it last imitated, and
// the expert builds on a sharper apprentice, so both strengths compound.
//
// The model tracks Elo-like strength for the apprentice and the expert across
// iterations. The expert's strength is the apprentice's strength plus a search
// gain, exactly as the paper frames it (the search improves on the apprentice's
// raw policy). The apprentice's next strength closes a fraction of the gap to
// the current expert, which is the imitation-learning convergence the paper
// describes. The two factors the paper names as controlling the learning rate,
// the size of the apprentice-to-improved-expert gap and how close the apprentice
// gets to its expert, are the two knobs of this recurrence.
//
// All randomness comes from the shared seeded rng threaded through the state, so
// a given seed and the same inputs reproduce the exact same run.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

// The paper reports MoHeX 1.0 (10,000 iterations per move) as the reference
// strength its agent must clear. Figure 3 places that line near this Elo.
export const MOHEX_ELO = 1100;

// Strength is bounded so the curves stay on a fixed chart no matter how the
// reader perturbs the run.
export const MAX_ELO = 1600;
export const MIN_ELO = -300;

export type TargetType = "TPT" | "CAT";

export type ExpertMode = "exit" | "reinforce";

export interface ExitParams {
  // The expert's tree-search budget. More simulations per move means a stronger
  // plan, with log-diminishing returns, matching MCTS's any-time behavior.
  simulations: number;
  // The weight w_a on the apprentice prior in the modified UCT formula. At 0 the
  // prior is ignored and the search is vanilla MCTS; the paper found ~100 best.
  priorWeight: number;
  // Whether the apprentice policy biases the search at all. Off recovers a
  // search that cannot use learned intuition.
  useApprenticePrior: boolean;
  // Whether a value network shortcuts rollouts. The paper's Figure 3 shows this
  // lifts both expert and apprentice once datasets are large enough.
  useValueNet: boolean;
  // Cost-sensitive tree-policy targets (TPT) train a stronger apprentice per
  // iteration than chosen-action targets (CAT), worth ~50 Elo in the paper.
  target: TargetType;
  // How many fresh expert positions feed the imitation step each iteration.
  // Larger datasets close more of the gap before overfitting.
  datasetSize: number;
  // ExIt bootstraps the expert with the apprentice. REINFORCE has no expert and
  // climbs by a noisy policy-gradient step instead.
  mode: ExpertMode;
}

export interface IterationPoint {
  iteration: number;
  apprentice: number;
  expert: number;
}

export interface ExitState {
  params: ExitParams;
  iteration: number;
  apprentice: number;
  expert: number;
  history: IterationPoint[];
  rngState: number;
  log: EventLog<ExitSnapshot>;
}

// The starting strength of an untrained network player.
const START_ELO = -120;

// The expert's search gain over the apprentice it builds on. Vanilla search
// (no prior) still helps, but the apprentice prior multiplies how far the search
// can reach, because it points the search at moves worth deep reading.
function searchGain(params: ExitParams): number {
  const depth = Math.log2(params.simulations / 100 + 1) * 120;

  // The prior helps most near the tuned weight (~100) and falls off if the
  // weight is far too small (ignored) or far too large (search trusts a weak
  // policy over its own reading). A simple bump centered at the paper's value.
  const tunedWeight = 100;
  const normalized = (params.priorWeight - tunedWeight) / 140;
  const weightFit = params.useApprenticePrior
    ? Math.exp(-(normalized * normalized))
    : 0;
  const priorBoost = 1 + 0.9 * weightFit;

  const valueBoost = params.useValueNet ? 1.25 : 1;

  return depth * priorBoost * valueBoost;
}

// The fraction of the apprentice-to-expert gap the imitation step closes in one
// iteration. TPT is cost-sensitive, so it gets the hard cases right and closes
// more of the gap than CAT for the same data. Bigger datasets close more, with
// diminishing returns as the network saturates.
function imitationRate(params: ExitParams): number {
  const targetQuality = params.target === "TPT" ? 0.62 : 0.5;
  const dataFactor = 1 - Math.exp(-params.datasetSize / 200000);
  return targetQuality * dataFactor;
}

function clampElo(value: number): number {
  return Math.min(MAX_ELO, Math.max(MIN_ELO, value));
}

// One iteration of the recurrence. The expert is the apprentice plus its search
// gain; the next apprentice closes a fraction of the gap up to that expert. A
// tiny seeded jitter stands in for run-to-run variance, which ExIt keeps small
// (the paper notes little variation between runs) but which we still thread
// through the rng so the run is reproducible and scrubbable.
function exitStep(
  apprentice: number,
  params: ExitParams,
  rngState: number,
): { apprentice: number; expert: number; rngState: number } {
  const gain = searchGain(params);
  const expert = clampElo(apprentice + gain);
  const rate = imitationRate(params);

  const draw = nextRandom(rngState);
  const jitter = (draw.value - 0.5) * 18;

  const nextApprentice = clampElo(
    apprentice + rate * (expert - apprentice) + jitter,
  );
  return { apprentice: nextApprentice, expert, rngState: draw.state };
}

// REINFORCE has no expert to imitate. It takes a small, high-variance policy
// gradient step that can regress, which is the instability the paper contrasts
// against ExIt. The expert strength equals the apprentice strength because there
// is no separate planner.
function reinforceStep(
  apprentice: number,
  rngState: number,
): { apprentice: number; expert: number; rngState: number } {
  const draw = nextRandom(rngState);
  const drift = 22;
  const noise = (draw.value - 0.5) * 150;
  const nextApprentice = clampElo(apprentice + drift + noise);
  return {
    apprentice: nextApprentice,
    expert: nextApprentice,
    rngState: draw.state,
  };
}

export function advance(state: ExitState): ExitState {
  const next =
    state.params.mode === "reinforce"
      ? reinforceStep(state.apprentice, state.rngState)
      : exitStep(state.apprentice, state.params, state.rngState);

  const iteration = state.iteration + 1;
  const point: IterationPoint = {
    iteration,
    apprentice: next.apprentice,
    expert: next.expert,
  };

  const label =
    state.params.mode === "reinforce"
      ? `iter ${iteration}: REINFORCE ${Math.round(next.apprentice)} Elo`
      : `iter ${iteration}: apprentice ${Math.round(next.apprentice)}, expert ${Math.round(next.expert)} Elo`;

  const advanced: ExitState = {
    ...state,
    iteration,
    apprentice: next.apprentice,
    expert: next.expert,
    history: [...state.history, point],
    rngState: next.rngState,
  };

  return {
    ...advanced,
    log: record(state.log, iteration, label, snapshot(advanced)),
  };
}

export type Intervention =
  | { type: "setSimulations"; value: number }
  | { type: "setPriorWeight"; value: number }
  | { type: "toggleApprenticePrior" }
  | { type: "toggleValueNet" }
  | { type: "setTarget"; value: TargetType }
  | { type: "setDatasetSize"; value: number }
  | { type: "setMode"; value: ExpertMode }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: ExitSnapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// A snapshot carries everything that defines a moment so the event log can
// rewind the whole run, not just the current parameters.
export interface ExitSnapshot {
  params: ExitParams;
  iteration: number;
  apprentice: number;
  expert: number;
  history: IterationPoint[];
  rngState: number;
}

export type ExitEvent = SimEvent<ExitSnapshot>;

function snapshot(state: ExitState): ExitSnapshot {
  return {
    params: { ...state.params },
    iteration: state.iteration,
    apprentice: state.apprentice,
    expert: state.expert,
    history: state.history.map((point) => ({ ...point })),
    rngState: state.rngState,
  };
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: ExitParams;
}

const EXIT_FULL: ExitParams = {
  simulations: 10000,
  priorWeight: 100,
  useApprenticePrior: true,
  useValueNet: false,
  target: "TPT",
  datasetSize: 243000,
  mode: "exit",
};

export const PRESETS: Preset[] = [
  {
    id: "exit",
    label: "ExIt (policy)",
    blurb:
      "The full loop with TPT targets and the tuned prior weight. Play it and watch the apprentice and expert climb past MoHeX together.",
    params: { ...EXIT_FULL },
  },
  {
    id: "value",
    label: "ExIt (policy + value)",
    blurb:
      "Add the value network. The expert reads positions without full rollouts, so both curves lift and clear MoHeX sooner, matching Figure 3.",
    params: { ...EXIT_FULL, useValueNet: true },
  },
  {
    id: "reinforce",
    label: "REINFORCE",
    blurb:
      "No expert, just a noisy policy gradient. The strength jitters and stalls low, the instability ExIt was built to beat.",
    params: { ...EXIT_FULL, mode: "reinforce" },
  },
  {
    id: "noprior",
    label: "Search without intuition",
    blurb:
      "Turn the apprentice prior off so the expert is vanilla MCTS. The search still helps, but it climbs slower because nothing points it at the moves worth reading deeply.",
    params: { ...EXIT_FULL, useApprenticePrior: false },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "exit", seed = 7): ExitState {
  const params = { ...getPreset(presetId).params };
  const expert =
    params.mode === "reinforce"
      ? START_ELO
      : clampElo(START_ELO + searchGain(params));
  return {
    params,
    iteration: 0,
    apprentice: START_ELO,
    expert,
    history: [{ iteration: 0, apprentice: START_ELO, expert }],
    rngState: seed >>> 0,
    log: emptyLog(),
  };
}

// Recompute the expert at the current apprentice strength after a parameter
// change, so the on-screen gap reacts the instant the reader flips a control,
// before the next iteration runs.
function rederiveExpert(state: ExitState): ExitState {
  const expert =
    state.params.mode === "reinforce"
      ? state.apprentice
      : clampElo(state.apprentice + searchGain(state.params));
  const history = state.history.map((point, index) =>
    index === state.history.length - 1 ? { ...point, expert } : point,
  );
  return { ...state, expert, history };
}

function applyIntervention(
  state: ExitState,
  action: Intervention,
): { state: ExitState; label: string } {
  switch (action.type) {
    case "setSimulations": {
      const params = { ...state.params, simulations: action.value };
      return {
        state: rederiveExpert({ ...state, params }),
        label: `simulations -> ${action.value}`,
      };
    }
    case "setPriorWeight": {
      const params = { ...state.params, priorWeight: action.value };
      return {
        state: rederiveExpert({ ...state, params }),
        label: `prior weight -> ${action.value}`,
      };
    }
    case "toggleApprenticePrior": {
      const useApprenticePrior = !state.params.useApprenticePrior;
      const params = { ...state.params, useApprenticePrior };
      return {
        state: rederiveExpert({ ...state, params }),
        label: `apprentice prior ${useApprenticePrior ? "on" : "off"}`,
      };
    }
    case "toggleValueNet": {
      const useValueNet = !state.params.useValueNet;
      const params = { ...state.params, useValueNet };
      return {
        state: rederiveExpert({ ...state, params }),
        label: `value net ${useValueNet ? "on" : "off"}`,
      };
    }
    case "setTarget": {
      const params = { ...state.params, target: action.value };
      return {
        state: { ...state, params },
        label: `target -> ${action.value}`,
      };
    }
    case "setDatasetSize": {
      const params = { ...state.params, datasetSize: action.value };
      return {
        state: { ...state, params },
        label: `dataset -> ${action.value}`,
      };
    }
    case "setMode": {
      const params = { ...state.params, mode: action.value };
      return {
        state: rederiveExpert({ ...state, params }),
        label: `mode -> ${action.value === "exit" ? "ExIt" : "REINFORCE"}`,
      };
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      const fresh = initialState(preset.id, state.rngState);
      return {
        state: { ...fresh, log: state.log },
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore": {
      const restored: ExitState = {
        ...state,
        params: { ...action.snapshot.params },
        iteration: action.snapshot.iteration,
        apprentice: action.snapshot.apprentice,
        expert: action.snapshot.expert,
        history: action.snapshot.history.map((point) => ({ ...point })),
        rngState: action.snapshot.rngState,
      };
      return { state: restored, label: action.label };
    }
  }
}

// The pure reducer. A tick runs one ExIt (or REINFORCE) iteration; an
// intervention changes a parameter and records the moment. Every step is
// deterministic given the state's rng cursor.
export function step(state: ExitState, input: Input): ExitState {
  if (input.kind === "tick") return advance(state);
  const { state: next, label } = applyIntervention(state, input.action);
  return {
    ...next,
    log: record(next.log, next.iteration, label, snapshot(next)),
  };
}

export interface ExitReadout {
  gap: number;
  beatsMohex: boolean;
  iterationsToMohex: number | null;
  searchGain: number;
}

// The quantities worth reading off the run: the live apprentice-to-expert gap,
// whether the apprentice has cleared MoHeX, and when it first did so.
export function readout(state: ExitState): ExitReadout {
  const crossing = state.history.find((point) => point.apprentice >= MOHEX_ELO);
  return {
    gap: state.expert - state.apprentice,
    beatsMohex: state.apprentice >= MOHEX_ELO,
    iterationsToMohex: crossing ? crossing.iteration : null,
    searchGain:
      state.params.mode === "reinforce" ? 0 : searchGain(state.params),
  };
}
