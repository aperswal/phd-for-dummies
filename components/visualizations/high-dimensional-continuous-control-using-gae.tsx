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
  type ValueMode,
} from "@/components/visualizations/high-dimensional-continuous-control-using-gae-model";

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

const VALUE_LABELS: Record<ValueMode, string> = {
  exact: "exact V",
  noisy: "noisy V",
  zero: "zero V",
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

// Two stacked polylines: expected return against the optimal ceiling, and the
// estimator's bias over the same steps, so the reader watches the climb and the
// bias that drives it at once.
function CurvePair({
  rho,
  bias,
  optimal,
}: {
  rho: number[];
  bias: number[];
  optimal: number;
}) {
  if (rho.length < 2) {
    return (
      <p className="text-muted-foreground text-xs">
        Run a few steps to draw the return and bias curves.
      </p>
    );
  }
  const lo = Math.min(0, ...rho);
  const hi = Math.max(optimal, ...rho);
  const span = hi - lo < 1e-9 ? 1 : hi - lo;
  const toY = (value: number) => 24 - ((value - lo) / span) * 22 - 1;
  const rhoPoints = rho
    .map((value, index) => {
      const x = (index / (rho.length - 1)) * 100;
      return `${x.toFixed(1)},${toY(value).toFixed(1)}`;
    })
    .join(" ");
  const biasPoints = bias
    .map((value, index) => {
      const x = (index / (bias.length - 1)) * 100;
      const y = 24 - Math.min(1, value) * 22 - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox="0 0 100 24"
      className="h-12 w-full"
      role="img"
      aria-label="Expected return climbing toward the optimal ceiling, with the estimator's bias drawn underneath"
    >
      <line
        x1={0}
        y1={toY(optimal)}
        x2={100}
        y2={toY(optimal)}
        stroke={SAGE}
        strokeWidth={0.8}
        strokeDasharray="2 2"
      />
      <polyline
        points={biasPoints}
        fill="none"
        stroke={SLATE}
        strokeWidth={1}
        strokeDasharray="1.5 1.5"
      />
      <polyline
        points={rhoPoints}
        fill="none"
        stroke={ACCENT}
        strokeWidth={1.4}
      />
    </svg>
  );
}

function cellCenter(state: number): { cx: number; cy: number } {
  return {
    cx: colOf(state) * CELL + CELL / 2,
    cy: rowOf(state) * CELL + CELL / 2,
  };
}

export function GeneralizedAdvantageEstimation() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("balanced"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [presetId, setPresetId] = useState("balanced");
  const [inspect, setInspect] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { mdp, params, pi, log } = state;
  const score = useMemo(() => status(state), [state]);

  const maxVisitProxy = useMemo(() => {
    // Use the discounted-return scale of the true value as a shading proxy so the
    // grid lights along the path the policy favors.
    let max = 1e-9;
    for (let s = 0; s < state.trueValues.length; s++) {
      if (isTerminal(mdp, s)) continue;
      max = Math.max(max, Math.abs(state.trueValues[s] ?? 0));
    }
    return max;
  }, [state.trueValues, mdp]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 260 / speed, () =>
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

  const trajectory = state.lastTrajectory;
  const trajectoryPath =
    trajectory && trajectory.length > 1
      ? trajectory
          .map((s) => {
            const { cx, cy } = cellCenter(s);
            return `${cx},${cy}`;
          })
          .join(" ")
      : null;

  const inspected =
    inspect !== null && !isTerminal(mdp, inspect) ? inspect : null;

  const biasVerdict = score.unbiased
    ? { label: `unbiased (${score.bias.toFixed(2)})`, color: SAGE }
    : { label: `biased (${score.bias.toFixed(2)})`, color: ACCENT };

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
      <p className="text-muted-foreground text-sm">
        {PRESETS.find((preset) => preset.id === presetId)?.blurb}
      </p>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            ref={svgRef}
            viewBox={`-2 -2 ${VIEW + 4} ${VIEW + 4}`}
            className="w-full cursor-pointer touch-none"
            role="img"
            aria-label={`A gridworld policy learned with generalized advantage estimation. Arrows show the policy's favored move in each cell. Expected return is ${pct(score.fractionOfOptimal)} of optimal, the estimator's bias is ${score.bias.toFixed(2)}, and its variance is ${score.variance.toFixed(2)}.`}
            onClick={onCanvasClick}
          >
            {/* Value shading: sage where the solved value is high, the region the
                policy is learning to favor. */}
            {state.trueValues.map((v, s) => {
              if (isTerminal(mdp, s)) return null;
              return (
                <rect
                  key={`bg-${s}`}
                  x={colOf(s) * CELL}
                  y={rowOf(s) * CELL}
                  width={CELL}
                  height={CELL}
                  fill={SAGE}
                  fillOpacity={0.06 + 0.5 * Math.max(0, v / maxVisitProxy)}
                  stroke="var(--color-foreground)"
                  strokeOpacity={0.08}
                  strokeWidth={0.4}
                />
              );
            })}

            {/* The most recent sampled episode path. */}
            {trajectoryPath && (
              <polyline
                points={trajectoryPath}
                fill="none"
                stroke={SLATE}
                strokeOpacity={0.7}
                strokeWidth={0.8}
                strokeDasharray="1.5 1.5"
                strokeLinejoin="round"
              />
            )}

            {/* Policy arrows: length is the action's probability, the most likely
                one drawn in clay-orange. */}
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
                    const len = 2 + p * (CELL / 2 - 2.5);
                    return (
                      <line
                        key={a}
                        x1={cx}
                        y1={cy}
                        x2={cx + dir.x * len}
                        y2={cy + dir.y * len}
                        stroke={a === best ? ACCENT : "var(--color-foreground)"}
                        strokeOpacity={a === best ? 0.95 : 0.35}
                        strokeWidth={a === best ? 1.5 : 0.8}
                        strokeLinecap="round"
                      />
                    );
                  })}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={0.9}
                    fill="var(--color-foreground)"
                    fillOpacity={0.5}
                  />
                </g>
              );
            })}

            {/* Goal (sage, reward +1), trap (slate, reward -1), start ring. */}
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

        <SimPanel title="What the estimator is doing" className="lg:self-start">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">step {state.steps}</Badge>
            <Badge variant="outline">
              return {pct(score.fractionOfOptimal)}
            </Badge>
            <Badge variant="outline">{VALUE_LABELS[params.valueMode]}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              variant="default"
              style={{ backgroundColor: biasVerdict.color, color: "#fff" }}
            >
              bias {biasVerdict.label}
            </Badge>
            <Badge variant="outline">
              variance {score.variance.toFixed(2)}
            </Badge>
            <Badge
              variant={score.nearOptimal ? "default" : "outline"}
              style={
                score.nearOptimal
                  ? { backgroundColor: SAGE, color: "#fff" }
                  : undefined
              }
            >
              {score.nearOptimal ? "near optimal" : "climbing"}
            </Badge>
          </div>

          <CurvePair
            rho={state.rhoHistory}
            bias={state.biasHistory}
            optimal={score.optimal}
          />

          {inspected !== null ? (
            <div className="text-muted-foreground flex flex-col gap-1 text-xs">
              <span>
                cell ({colOf(inspected)}, {rowOf(inspected)}), value{" "}
                <span className="text-foreground font-mono">
                  {(state.estValue[inspected] ?? 0).toFixed(2)}
                </span>
              </span>
              {ACTION_LABELS.map((label, a) => {
                const adv = state.trueAdvantage[inspected]?.[a] ?? 0;
                return (
                  <span key={a} className="font-mono">
                    {label.padEnd(6)} p={(pi[inspected]?.[a] ?? 0).toFixed(2)}{" "}
                    A=
                    <span style={{ color: adv >= 0 ? SAGE : SLATE }}>
                      {adv >= 0 ? "+" : ""}
                      {adv.toFixed(2)}
                    </span>
                  </span>
                );
              })}
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click a cell to read its value estimate, its move probabilities,
              and each move&apos;s true advantage. The longest arrow is the
              policy&apos;s favored move; clay-orange marks it. Greener cells
              hold more value.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Reach optimal or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.tick}`,
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
        />
        <div className="min-w-0 w-40">
          <Slider
            label="Playback speed (animation pace only, not part of GAE)"
            value={speed}
            min={1}
            max={12}
            step={1}
            display={`${speed}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <SimPanel title="Knobs">
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="λ — slides between one-step biased (0) and full-return unbiased (1)"
            value={params.lambda}
            min={0}
            max={1}
            step={0.01}
            display={params.lambda.toFixed(2)}
            onChange={(value) => intervene({ type: "setLambda", value })}
          />
          <Slider
            label="γ — discount on future rewards; defines which value function we solve"
            value={params.gamma}
            min={0.9}
            max={0.995}
            step={0.005}
            display={params.gamma.toFixed(3)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">value function</span>
          {(
            [
              ["exact", "exact"],
              ["noisy", "noisy"],
              ["zero", "zero"],
            ] as [ValueMode, string][]
          ).map(([mode, label]) => (
            <Toggle
              key={mode}
              label={label}
              active={params.valueMode === mode}
              onClick={() => intervene({ type: "setValueMode", value: mode })}
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
            label="value error — std of per-state noise added to V (noisy mode only)"
            value={params.valueError}
            min={0}
            max={1}
            step={0.05}
            display={params.valueError.toFixed(2)}
            disabled={params.valueMode !== "noisy"}
            onChange={(value) => intervene({ type: "setValueError", value })}
          />
          <Slider
            label="episodes per step — batch size; more reduces variance of the gradient estimate"
            value={params.episodes}
            min={2}
            max={80}
            step={2}
            display={String(params.episodes)}
            onChange={(value) => intervene({ type: "setEpisodes", value })}
          />
          <Slider
            label="α — policy-gradient step size; larger climbs faster but can overshoot"
            value={params.alpha}
            min={0.5}
            max={8}
            step={0.1}
            display={params.alpha.toFixed(1)}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
        </div>
      </SimPanel>

      <p className="text-muted-foreground text-xs">
        A softmax policy on a 5 by 5 gridworld with a goal worth +1 and a trap
        worth &minus;1. Each step samples a batch of episodes, estimates the
        advantage at every visited step with GAE(gamma, lambda) =&nbsp;the
        exponentially-weighted sum of TD residuals, and takes one
        policy-gradient ascent step. The bias readout is the angle between the
        estimator&apos;s expected gradient and the true gradient, computed on
        the side only to score it. It sits near zero at lambda 1 for any value
        function, and grows as lambda shrinks under a wrong value function. This
        stands in for the paper&apos;s trust-region update and neural-network
        value function with one plain ascent step on a solvable grid. Each
        preset starts from a fixed rollout seed so the same preset always
        produces the same run; loading a preset resets it.
      </p>
    </div>
  );
}
