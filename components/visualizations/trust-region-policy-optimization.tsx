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
  colOf,
  GRID_W,
  initialState,
  isTerminal,
  N_ACTIONS,
  PRESETS,
  rowOf,
  status,
  step,
  type Intervention,
  type SimState,
  type UpdateMode,
} from "@/components/visualizations/trust-region-policy-optimization-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
const SLATE = "#7c8a99";
const CELL = 20;
const VIEW = GRID_W * CELL;

const ACTION_DIR: { x: number; y: number }[] = [
  { x: 0, y: -1 }, // up
  { x: 0, y: 1 }, // down
  { x: -1, y: 0 }, // left
  { x: 1, y: 0 }, // right
];

const UPDATE_LABELS: Record<UpdateMode, string> = {
  trustRegion: "trust region",
  penalty: "KL penalty",
  unconstrained: "no trust region",
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
  onChange: (value: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step: stepValue,
  display,
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
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer"
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

// The true return eta (sage) and the surrogate-predicted return (clay-orange)
// over rounds, both as a fraction of optimal under one dashed ceiling. With a
// trust region the two stay locked together and climb; without one the surrogate
// keeps rising while the true return lurches and drops below it.
function PerformanceSparkline({
  etaHistory,
  surrHistory,
  optimal,
}: {
  etaHistory: number[];
  surrHistory: number[];
  optimal: number;
}) {
  if (etaHistory.length < 2) {
    return (
      <p className="text-muted-foreground text-xs">
        Run a few rounds to draw the climb.
      </p>
    );
  }
  const denom = Math.abs(optimal) > 1e-9 ? optimal : 1;
  const etaFrac = etaHistory.map((v) => v / denom);
  const surrFrac = surrHistory.map((v) => v / denom);
  const lo = Math.min(0, ...etaFrac, ...surrFrac);
  const hi = Math.max(1, ...etaFrac, ...surrFrac);
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
      aria-label="True return and surrogate-predicted return as a fraction of optimal, per round"
    >
      <line
        x1={0}
        y1={toY(1)}
        x2={100}
        y2={toY(1)}
        stroke={SLATE}
        strokeWidth={0.7}
        strokeDasharray="2 2"
      />
      <polyline
        points={line(surrFrac)}
        fill="none"
        stroke={ACCENT}
        strokeWidth={1.1}
        strokeDasharray="3 2"
      />
      <polyline
        points={line(etaFrac)}
        fill="none"
        stroke={SAGE}
        strokeWidth={1.4}
      />
    </svg>
  );
}

// The trust-region meter: the step's measured mean KL as a bar against the delta
// budget. Inside the budget reads sage, a step that breaks it (only possible
// without a trust region) reads clay-orange.
function KLMeter({ kl, delta }: { kl: number; delta: number }) {
  const ceiling = Math.max(delta * 2.2, kl * 1.1, 0.04);
  const klFrac = Math.min(1, kl / ceiling);
  const deltaFrac = Math.min(1, delta / ceiling);
  const over = kl > delta + 1e-9;
  return (
    <div className="flex flex-col gap-1">
      <div className="ring-foreground/10 relative h-3 w-full overflow-hidden rounded ring-1">
        <div
          className="absolute inset-y-0 left-0"
          style={{
            width: `${klFrac * 100}%`,
            backgroundColor: over ? ACCENT : SAGE,
            opacity: 0.8,
          }}
        />
        <div
          className="absolute inset-y-0 w-px"
          style={{
            left: `${deltaFrac * 100}%`,
            backgroundColor: "var(--color-foreground)",
          }}
        />
      </div>
      <span className="text-muted-foreground text-xs">
        step KL{" "}
        <span className="text-foreground font-mono">{kl.toFixed(3)}</span> vs
        budget delta{" "}
        <span className="text-foreground font-mono">{delta.toFixed(3)}</span>
        {over ? " (over budget)" : ""}
      </span>
    </div>
  );
}

export function TrustRegionFlow() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("trust-region"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const [presetId, setPresetId] = useState("trust-region");
  const [inspect, setInspect] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { mdp, params, pi, rho } = state;
  const score = useMemo(() => status(state), [state]);

  const maxRho = useMemo(() => {
    let max = 1e-9;
    for (let s = 0; s < rho.length; s++) {
      if (isTerminal(mdp, s)) continue;
      max = Math.max(max, rho[s] ?? 0);
    }
    return max;
  }, [rho, mdp]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !state.done, 320 / speed, () =>
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
            aria-label={`A gridworld policy under trust region policy optimization. Cells are shaded by how often the policy visits them from the start. Arrows are the current move probabilities. The true return is ${pct(score.fraction)} of optimal.`}
            onClick={onCanvasClick}
          >
            {/* Visitation shading: where the policy spends time from the start. */}
            {rho.map((d, s) => {
              if (isTerminal(mdp, s)) return null;
              return (
                <rect
                  key={`bg-${s}`}
                  x={colOf(s) * CELL}
                  y={rowOf(s) * CELL}
                  width={CELL}
                  height={CELL}
                  fill={SAGE}
                  fillOpacity={0.06 + 0.5 * (d / maxRho)}
                  stroke="var(--color-foreground)"
                  strokeOpacity={0.08}
                  strokeWidth={0.4}
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
            <Badge variant="outline">true {pct(score.fraction)}</Badge>
            <Badge variant="outline">{UPDATE_LABELS[params.update]}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              variant={score.overshot ? "default" : "outline"}
              style={
                score.overshot
                  ? { backgroundColor: ACCENT, color: "#fff" }
                  : undefined
              }
            >
              {score.overshot ? "overshot: surrogate lied" : "surrogate honest"}
            </Badge>
            <Badge
              variant={state.done ? "default" : "outline"}
              style={
                state.done
                  ? { backgroundColor: SAGE, color: "#fff" }
                  : undefined
              }
            >
              {state.done ? "settled" : "improving"}
            </Badge>
            {state.backtracks > 0 && (
              <Badge variant="outline">backtracked {state.backtracks}x</Badge>
            )}
          </div>

          <PerformanceSparkline
            etaHistory={state.etaHistory}
            surrHistory={state.surrHistory}
            optimal={score.optimal}
          />
          <p className="text-muted-foreground text-xs">
            Sage is the true return from the start, clay-orange dashed is what
            the surrogate predicts. The dashed ceiling is optimal. When they
            split, the policy moved past where the surrogate is trustworthy.
          </p>

          <KLMeter kl={score.appliedKL} delta={score.delta} />

          {inspected !== null ? (
            <div className="text-muted-foreground flex flex-col gap-1 text-xs">
              <span>
                cell ({colOf(inspected)}, {rowOf(inspected)}), visited{" "}
                <span className="text-foreground font-mono">
                  {(rho[inspected] ?? 0).toFixed(3)}
                </span>
              </span>
              {ACTION_LABELS.map((label, a) => {
                const adv = state.advantage[inspected]?.[a] ?? 0;
                const prob = pi[inspected]?.[a] ?? 0;
                return (
                  <span key={a} className="font-mono">
                    {label.padEnd(6)} p={prob.toFixed(2)} A=
                    <span style={{ color: adv >= 0 ? SAGE : SLATE }}>
                      {adv >= 0 ? "+" : ""}
                      {adv.toFixed(2)}
                    </span>
                  </span>
                );
              })}
              <span>p is the move probability, A its true advantage.</span>
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click a cell to read its move probabilities and each move&apos;s
              true advantage. Solid clay-orange is the policy&apos;s favored
              move.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Overshoots, backtracks, and control changes show up here."
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

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={state.done && !playing}
        />
        <div className="w-40">
          <Slider
            label="Speed"
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
              ["trustRegion", "trust region"],
              ["unconstrained", "no trust region"],
              ["penalty", "KL penalty"],
            ] as [UpdateMode, string][]
          ).map(([mode, label]) => (
            <Toggle
              key={mode}
              label={label}
              active={params.update === mode}
              onClick={() => intervene({ type: "setUpdate", value: mode })}
            />
          ))}
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={() => intervene({ type: "resetPolicy" })}
          >
            Reset policy
          </Button>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="delta (trust-region radius)"
            value={params.delta}
            min={0.005}
            max={0.6}
            step={0.005}
            display={params.delta.toFixed(3)}
            onChange={(value) => intervene({ type: "setDelta", value })}
          />
          <Slider
            label="advantage estimate error"
            value={params.estError}
            min={0}
            max={0.4}
            step={0.01}
            display={params.estError.toFixed(2)}
            onChange={(value) => intervene({ type: "setEstError", value })}
          />
          <Slider
            label="gamma (discount)"
            value={params.gamma}
            min={0.85}
            max={0.97}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
          <Slider
            label="seed (error pattern)"
            value={state.seed}
            min={1}
            max={40}
            step={1}
            display={String(state.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </div>
      </SimPanel>

      <p className="text-muted-foreground text-xs">
        The policy is a softmax over moves in each cell of a 5 by 5 gridworld
        with a goal worth +1 and a trap worth &minus;1. Every round the model
        solves the policy exactly for its value, advantage, and discounted
        visitation, builds the surrogate and its natural-gradient step, then
        scales the step to a mean KL of delta and backtracks until the surrogate
        truly improves. The advantage estimate carries a fixed seeded error, so
        a worse estimate makes the surrogate less trustworthy. This runs one
        small MDP solved exactly, where the paper estimates these quantities
        from sampled trajectories.
      </p>
    </div>
  );
}
