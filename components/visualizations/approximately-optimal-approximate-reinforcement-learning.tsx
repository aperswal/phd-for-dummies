"use client";

import { useCallback, useMemo, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  ACTION_LABELS,
  actualImprovement,
  boundZeroCrossing,
  chosenAlpha,
  colOf,
  GRID_W,
  improvementBound,
  initialState,
  isTerminal,
  N_ACTIONS,
  PRESETS,
  rowOf,
  safeStep,
  status,
  step,
  type Intervention,
  type RestartMode,
  type SimState,
  type UpdateMode,
} from "@/components/visualizations/approximately-optimal-approximate-reinforcement-learning-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
const SLATE = "#7c8a99";
const CELL = 20;
const VIEW = GRID_W * CELL;
const CORNERS = [0, 4, 20, 24];

const ACTION_DIR: { x: number; y: number }[] = [
  { x: 0, y: -1 }, // up
  { x: 0, y: 1 }, // down
  { x: -1, y: 0 }, // left
  { x: 1, y: 0 }, // right
];

const UPDATE_LABELS: Record<UpdateMode, string> = {
  conservative: "conservative",
  greedy: "greedy (alpha = 1)",
  manual: "manual alpha",
};

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function cellCenter(state: number): { cx: number; cy: number } {
  return {
    cx: colOf(state) * CELL + CELL / 2,
    cy: rowOf(state) * CELL + CELL / 2,
  };
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step: stepValue,
  display,
  disabled,
  onChange,
}: SliderProps) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-muted-foreground flex items-center justify-between">
        <span>{label}</span>
        <span className="text-foreground font-mono">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={stepValue}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
        style={{ accentColor: ACCENT }}
      />
    </label>
  );
}

function Toggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant={active ? "default" : "outline"}
      aria-pressed={active}
      onClick={onClick}
    >
      {label}
    </Button>
  );
}

// Two performance curves over rounds, each as a fraction of its own optimal so
// they share a 0-to-1 axis under a single dashed ceiling: clay-orange for the
// restart objective eta_mu the algorithm improves, slate for the true objective
// eta_D from the start state. The reader watches the conservative curve rise
// monotonically and the greedy one lurch.
function PerformanceSparkline({
  muHistory,
  dHistory,
  optimalMu,
  optimalD,
}: {
  muHistory: number[];
  dHistory: number[];
  optimalMu: number;
  optimalD: number;
}) {
  if (muHistory.length < 2) {
    return (
      <p className="text-muted-foreground text-xs">
        Run a few rounds to draw the performance curves.
      </p>
    );
  }
  const muFrac = muHistory.map((v) => (optimalMu > 1e-9 ? v / optimalMu : 0));
  const dFrac = dHistory.map((v) => (optimalD > 1e-9 ? v / optimalD : 0));
  const lo = Math.min(0, ...muFrac, ...dFrac);
  const hi = Math.max(1, ...muFrac, ...dFrac);
  const span = hi - lo < 1e-9 ? 1 : hi - lo;
  const toY = (value: number) => 24 - ((value - lo) / span) * 22 - 1;
  const line = (series: number[]) =>
    series
      .map((value, index) => {
        const x = (index / (series.length - 1)) * 100;
        return `${x.toFixed(1)},${toY(value).toFixed(1)}`;
      })
      .join(" ");
  return (
    <svg
      viewBox="0 0 100 24"
      className="h-12 w-full"
      role="img"
      aria-label="Performance as a fraction of optimal per round, for the restart objective and the true start objective"
    >
      <line
        x1={0}
        y1={toY(1)}
        x2={100}
        y2={toY(1)}
        stroke={SAGE}
        strokeWidth={0.7}
        strokeDasharray="2 2"
      />
      <polyline
        points={line(dFrac)}
        fill="none"
        stroke={SLATE}
        strokeWidth={1.1}
      />
      <polyline
        points={line(muFrac)}
        fill="none"
        stroke={ACCENT}
        strokeWidth={1.4}
      />
    </svg>
  );
}

// Theorem 4.1's guaranteed gain (clay-orange) and the actual change in eta_mu
// (sage) as functions of the mixture step alpha, zoomed to the window where the
// guarantee is interesting: it rises to the safe step, crosses zero, and goes
// negative. The dashed vertical line marks the safe step the conservative update
// takes; greedy commits at alpha = 1, far to the right where the guarantee is
// deeply negative.
function BoundChart({ state }: { state: SimState }) {
  const curve = useMemo(() => {
    const zero = boundZeroCrossing(state);
    const alphaMax = zero <= 1e-6 ? 0.02 : Math.min(1, zero * 1.6);
    const points = 28;
    const bounds: number[] = [];
    const actuals: number[] = [];
    const alphas: number[] = [];
    for (let i = 0; i <= points; i++) {
      const alpha = (i / points) * alphaMax;
      alphas.push(alpha);
      bounds.push(improvementBound(state, alpha));
      actuals.push(actualImprovement(state, alpha));
    }
    return { alphaMax, alphas, bounds, actuals, safe: safeStep(state) };
  }, [state]);

  if (curve.alphaMax <= 1e-6 || state.policyAdvantage <= 0) {
    return (
      <p className="text-muted-foreground text-xs">
        No improving step left. The policy advantage has reached the break
        point, so the guaranteed-gain curve is flat at zero.
      </p>
    );
  }

  const lo = Math.min(0, ...curve.bounds, ...curve.actuals);
  const hi = Math.max(0, ...curve.bounds, ...curve.actuals);
  const span = hi - lo < 1e-9 ? 1 : hi - lo;
  const toX = (alpha: number) => (alpha / curve.alphaMax) * 100;
  const toY = (value: number) => 30 - ((value - lo) / span) * 28 - 1;
  const points = (series: number[]) =>
    series
      .map(
        (value, index) =>
          `${toX(curve.alphas[index] ?? 0).toFixed(1)},${toY(value).toFixed(1)}`,
      )
      .join(" ");

  return (
    <svg
      viewBox="0 0 100 30"
      className="h-20 w-full"
      role="img"
      aria-label="Guaranteed and actual change in performance as a function of the mixture step size"
    >
      <line
        x1={0}
        y1={toY(0)}
        x2={100}
        y2={toY(0)}
        stroke="var(--color-foreground)"
        strokeOpacity={0.25}
        strokeWidth={0.5}
      />
      <line
        x1={toX(curve.safe)}
        y1={0}
        x2={toX(curve.safe)}
        y2={30}
        stroke={SLATE}
        strokeWidth={0.6}
        strokeDasharray="2 1.5"
      />
      <polyline
        points={points(curve.actuals)}
        fill="none"
        stroke={SAGE}
        strokeWidth={1.1}
      />
      <polyline
        points={points(curve.bounds)}
        fill="none"
        stroke={ACCENT}
        strokeWidth={1.4}
      />
    </svg>
  );
}

export function ConservativePolicyIteration() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("conservative"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [presetId, setPresetId] = useState("conservative");
  const [inspect, setInspect] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { mdp, params, pi, visitMu, greedy, log } = state;
  const score = useMemo(() => status(state), [state]);
  const nextAlpha = useMemo(() => chosenAlpha(state), [state]);

  const maxVisit = useMemo(() => {
    let max = 1e-9;
    for (let s = 0; s < visitMu.length; s++) {
      if (isTerminal(mdp, s)) continue;
      max = Math.max(max, visitMu[s] ?? 0);
    }
    return max;
  }, [visitMu, mdp]);

  const restartCells = useMemo(() => {
    if (params.restart === "start") return [mdp.start];
    if (params.restart === "corners")
      return CORNERS.filter((c) => !isTerminal(mdp, c));
    return [];
  }, [params.restart, mdp]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !state.done, 240 / speed, () =>
    dispatch({ kind: "tick" }),
  );

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    setInspect(null);
    dispatch({ kind: "reset", presetId });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      setInspect(null);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  const onCanvasClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const col = Math.floor(
        ((event.clientX - rect.left) / rect.width) * GRID_W,
      );
      const row = Math.floor(
        ((event.clientY - rect.top) / rect.height) * GRID_W,
      );
      const cell = row * GRID_W + col;
      setInspect((current) => (current === cell ? null : cell));
    },
    [],
  );

  const inspected =
    inspect !== null && !isTerminal(mdp, inspect) ? inspect : null;
  const mismatchDisplay = Number.isFinite(score.mismatch)
    ? score.mismatch.toFixed(2)
    : "infinite";

  const blurb = PRESETS.find((preset) => preset.id === presetId)?.blurb;

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((preset) => (
          <Button
            key={preset.id}
            type="button"
            size="sm"
            variant={presetId === preset.id ? "default" : "outline"}
            aria-pressed={presetId === preset.id}
            onClick={() => loadPreset(preset.id)}
          >
            {preset.label}
          </Button>
        ))}
      </div>
      <p className="text-muted-foreground text-sm">{blurb}</p>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            ref={svgRef}
            viewBox={`-2 -2 ${VIEW + 4} ${VIEW + 4}`}
            className="w-full cursor-pointer touch-none"
            role="img"
            aria-label={`A gridworld policy under conservative policy iteration. Cells are shaded by how often the restart distribution visits them. Solid arrows are the current policy, hollow chevrons are the greedy chooser's proposal. Performance is ${pct(score.fractionMu)} of optimal under the restart objective.`}
            onClick={onCanvasClick}
          >
            {/* Visitation shading: where the policy spends time under mu. */}
            {visitMu.map((d, s) => {
              if (isTerminal(mdp, s)) return null;
              return (
                <rect
                  key={`bg-${s}`}
                  x={colOf(s) * CELL}
                  y={rowOf(s) * CELL}
                  width={CELL}
                  height={CELL}
                  fill={SAGE}
                  fillOpacity={0.06 + 0.5 * (d / maxVisit)}
                  stroke="var(--color-foreground)"
                  strokeOpacity={0.08}
                  strokeWidth={0.4}
                />
              );
            })}

            {/* Restart cells when mu is not uniform: where practice begins. */}
            {restartCells.map((s) => (
              <rect
                key={`mu-${s}`}
                x={colOf(s) * CELL + 1.5}
                y={rowOf(s) * CELL + 1.5}
                width={CELL - 3}
                height={CELL - 3}
                rx={2}
                fill="none"
                stroke={ACCENT}
                strokeOpacity={0.55}
                strokeWidth={0.7}
                strokeDasharray="1.5 1.5"
              />
            ))}

            {/* Greedy proposal pi': a hollow chevron toward the chooser's action. */}
            {greedy.map((g, s) => {
              if (g < 0 || isTerminal(mdp, s)) return null;
              const { cx, cy } = cellCenter(s);
              const dir = ACTION_DIR[g] ?? { x: 0, y: 0 };
              const tipX = cx + dir.x * (CELL / 2 - 2);
              const tipY = cy + dir.y * (CELL / 2 - 2);
              const px = -dir.y;
              const py = dir.x;
              return (
                <path
                  key={`g-${s}`}
                  d={`M ${tipX - dir.x * 2 + px * 1.6} ${tipY - dir.y * 2 + py * 1.6} L ${tipX} ${tipY} L ${tipX - dir.x * 2 - px * 1.6} ${tipY - dir.y * 2 - py * 1.6}`}
                  fill="none"
                  stroke={SLATE}
                  strokeOpacity={0.8}
                  strokeWidth={0.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              );
            })}

            {/* Current policy: arrow length is the move's probability, the most
                likely move drawn in clay-orange. */}
            {pi.map((probs, s) => {
              if (isTerminal(mdp, s)) return null;
              const { cx, cy } = cellCenter(s);
              let best = 0;
              for (let a = 1; a < N_ACTIONS; a++) {
                if ((probs[a] ?? 0) > (probs[best] ?? 0)) best = a;
              }
              return (
                <g key={`pi-${s}`}>
                  {probs.map((p, a) => {
                    const dir = ACTION_DIR[a] ?? { x: 0, y: 0 };
                    const len = 1.5 + p * (CELL / 2 - 3);
                    return (
                      <line
                        key={a}
                        x1={cx}
                        y1={cy}
                        x2={cx + dir.x * len}
                        y2={cy + dir.y * len}
                        stroke={a === best ? ACCENT : "var(--color-foreground)"}
                        strokeOpacity={a === best ? 0.95 : 0.3}
                        strokeWidth={a === best ? 1.5 : 0.7}
                        strokeLinecap="round"
                      />
                    );
                  })}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={0.8}
                    fill="var(--color-foreground)"
                    fillOpacity={0.5}
                  />
                </g>
              );
            })}

            {/* Goal (sage, +1), trap (slate, -1), start ring. */}
            {(() => {
              const goal = cellCenter(mdp.goal);
              const trap = cellCenter(mdp.trap);
              const start = cellCenter(mdp.start);
              return (
                <g>
                  <rect
                    x={colOf(mdp.goal) * CELL + 2}
                    y={rowOf(mdp.goal) * CELL + 2}
                    width={CELL - 4}
                    height={CELL - 4}
                    rx={2}
                    fill={SAGE}
                    fillOpacity={0.85}
                  />
                  <path
                    d={`M ${goal.cx - 3} ${goal.cy} l 2 2.4 l 4 -5`}
                    fill="none"
                    stroke="#fff"
                    strokeWidth={1.3}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <rect
                    x={colOf(mdp.trap) * CELL + 2}
                    y={rowOf(mdp.trap) * CELL + 2}
                    width={CELL - 4}
                    height={CELL - 4}
                    rx={2}
                    fill={SLATE}
                    fillOpacity={0.75}
                  />
                  <path
                    d={`M ${trap.cx - 2.6} ${trap.cy - 2.6} l 5.2 5.2 M ${trap.cx + 2.6} ${trap.cy - 2.6} l -5.2 5.2`}
                    stroke="#fff"
                    strokeWidth={1.2}
                    strokeLinecap="round"
                  />
                  <circle
                    cx={start.cx}
                    cy={start.cy}
                    r={CELL / 2 - 1.5}
                    fill="none"
                    stroke="var(--color-foreground)"
                    strokeOpacity={0.5}
                    strokeWidth={0.8}
                    strokeDasharray="2 1.5"
                  />
                </g>
              );
            })()}

            {inspected !== null && (
              <rect
                x={colOf(inspected) * CELL + 0.5}
                y={rowOf(inspected) * CELL + 0.5}
                width={CELL - 1}
                height={CELL - 1}
                rx={2}
                fill="none"
                stroke={ACCENT}
                strokeWidth={1.2}
              />
            )}
          </svg>
        </div>

        <SimPanel title="Where the climb stands" className="lg:self-start">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">round {state.steps}</Badge>
            <Badge variant="outline">eta_mu {pct(score.fractionMu)}</Badge>
            <Badge variant="outline">eta_D {pct(score.fractionD)}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="outline">{UPDATE_LABELS[params.update]}</Badge>
            <Badge
              variant={Number.isFinite(score.mismatch) ? "outline" : "default"}
              style={
                Number.isFinite(score.mismatch)
                  ? undefined
                  : { backgroundColor: ACCENT, color: "#fff" }
              }
            >
              mismatch {mismatchDisplay}
            </Badge>
            <Badge
              variant={state.done ? "default" : "outline"}
              style={
                state.done
                  ? { backgroundColor: SAGE, color: "#fff" }
                  : undefined
              }
            >
              {state.done ? "stopped at break point" : "improving"}
            </Badge>
          </div>

          <PerformanceSparkline
            muHistory={state.etaMuHistory}
            dHistory={state.etaDHistory}
            optimalMu={score.optimalMu}
            optimalD={score.optimalD}
          />
          <p className="text-muted-foreground text-xs">
            Clay-orange is performance from the restart distribution (what the
            algorithm improves), slate is performance from the true start. The
            dashed line is optimal.
          </p>

          {inspected !== null ? (
            <div className="text-muted-foreground flex flex-col gap-1 text-xs">
              <span>
                cell ({colOf(inspected)}, {rowOf(inspected)}), visited{" "}
                <span className="text-foreground font-mono">
                  {(visitMu[inspected] ?? 0).toFixed(3)}
                </span>
              </span>
              {ACTION_LABELS.map((label, a) => {
                const adv = state.advantage[inspected]?.[a] ?? 0;
                const isGreedy = greedy[inspected] === a;
                return (
                  <span key={a} className="font-mono">
                    {isGreedy ? "*" : " "}
                    {label.padEnd(6)} p={(pi[inspected]?.[a] ?? 0).toFixed(2)}{" "}
                    A=
                    <span style={{ color: adv >= 0 ? SAGE : SLATE }}>
                      {adv >= 0 ? "+" : ""}
                      {adv.toFixed(2)}
                    </span>
                  </span>
                );
              })}
              <span>* marks the greedy chooser&apos;s proposed move.</span>
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click a cell to read its move probabilities and each move&apos;s
              advantage. Solid clay-orange is the policy&apos;s favored move,
              hollow slate chevrons are the greedy chooser&apos;s proposal.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Drops, stops, and control changes show up here."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to round ${event.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <SimPanel title="Guaranteed gain vs. actual gain by step size">
        <BoundChart state={state} />
        <p className="text-muted-foreground text-xs">
          Clay-orange is Theorem 4.1&apos;s guaranteed improvement, sage is the
          real change in performance, both against the mixture step. The dashed
          line is the safe step the conservative update takes. The next round
          will use alpha ={" "}
          <span className="text-foreground font-mono">
            {nextAlpha.toFixed(4)}
          </span>
          . Greedy sets alpha = 1, far past where the guarantee turns negative.
        </p>
      </SimPanel>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={state.done && !playing}
        />
        <div className="w-48">
          <Slider
            label="Playback pace (rounds per second)"
            value={speed}
            min={1}
            max={16}
            step={1}
            display={`${speed}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <SimPanel title="Knobs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">update</span>
          {(
            [
              ["conservative", "conservative"],
              ["greedy", "greedy (alpha = 1)"],
              ["manual", "manual"],
            ] as [UpdateMode, string][]
          ).map(([mode, label]) => (
            <Toggle
              key={mode}
              label={label}
              active={params.update === mode}
              onClick={() => intervene({ type: "setUpdate", value: mode })}
            />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">restart</span>
          {(
            [
              ["uniform", "uniform"],
              ["start", "start only"],
              ["corners", "corners"],
            ] as [RestartMode, string][]
          ).map(([mode, label]) => (
            <Toggle
              key={mode}
              label={label}
              active={params.restart === mode}
              onClick={() => intervene({ type: "setRestart", value: mode })}
            />
          ))}
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={() => intervene({ type: "resetPolicy" })}
          >
            Reset policy to uniform
          </Button>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="manual alpha — how far to move toward the greedy policy each round"
            value={params.alpha}
            min={0.01}
            max={1}
            step={0.01}
            display={params.alpha.toFixed(2)}
            disabled={params.update !== "manual"}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          <Slider
            label="chooser error — how noisy the value estimates are (epsilon)"
            value={params.chooserError}
            min={0}
            max={0.6}
            step={0.01}
            display={params.chooserError.toFixed(2)}
            onChange={(value) => intervene({ type: "setChooserError", value })}
          />
          <Slider
            label="gamma — discount factor, controls how much future reward counts"
            value={params.gamma}
            min={0.8}
            max={0.97}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
        </div>
      </SimPanel>

      <p className="text-muted-foreground text-xs">
        The policy is a distribution over moves in each cell of a 5 by 5
        gridworld with a goal worth +1 and a trap worth &minus;1. Every round
        the model solves the policy exactly for its value, advantage, and
        visitation under the restart distribution, asks an approximate greedy
        chooser for a proposal, then mixes toward it by the chosen step. The
        chooser sees value estimates carrying error that varies each round, so a
        worse chooser stops the run at a worse policy. The chooser&apos;s error
        pattern is fixed per preset (deterministic seed), so the same preset
        always produces the same run. This runs one small MDP, not thousands of
        states, so the visitation and advantages are exact rather than sampled.
      </p>
    </div>
  );
}
