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
  type CriticMode,
  type Intervention,
  type SimState,
} from "@/components/visualizations/policy-gradient-methods-model";

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

const CRITIC_LABELS: Record<CriticMode, string> = {
  exact: "true value",
  compatible: "compatible critic",
  incompatible: "incompatible critic",
  reinforce: "sampled returns",
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

// A short polyline of recent expected return, drawn against the optimal return
// as a dashed ceiling, so the reader watches the climb (or the stall) as a curve.
function ReturnSparkline({
  history,
  optimal,
}: {
  history: number[];
  optimal: number;
}) {
  if (history.length < 2) {
    return (
      <p className="text-muted-foreground text-xs">
        Run a few steps to draw the return curve.
      </p>
    );
  }
  const lo = Math.min(0, ...history);
  const hi = Math.max(optimal, ...history);
  const span = hi - lo < 1e-9 ? 1 : hi - lo;
  const toY = (value: number) => 24 - ((value - lo) / span) * 22 - 1;
  const points = history
    .map((value, index) => {
      const x = (index / (history.length - 1)) * 100;
      return `${x.toFixed(1)},${toY(value).toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox="0 0 100 24"
      className="h-10 w-full"
      role="img"
      aria-label="Expected return per step against the optimal"
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
      <polyline points={points} fill="none" stroke={ACCENT} strokeWidth={1.4} />
    </svg>
  );
}

function cellCenter(state: number): { cx: number; cy: number } {
  return {
    cx: colOf(state) * CELL + CELL / 2,
    cy: rowOf(state) * CELL + CELL / 2,
  };
}

export function PolicyGradientGrid() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("exact"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [presetId, setPresetId] = useState("exact");
  const [inspect, setInspect] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { mdp, params, pi, visitation, log } = state;
  const score = useMemo(() => status(state), [state]);

  const maxVisit = useMemo(() => {
    let max = 1e-9;
    for (let s = 0; s < visitation.length; s++) {
      if (isTerminal(mdp, s)) continue;
      max = Math.max(max, visitation[s] ?? 0);
    }
    return max;
  }, [visitation, mdp]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 240 / speed, () =>
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

  const verdict = (() => {
    if (params.critic === "reinforce")
      return { label: `noisy (${score.cos.toFixed(2)})`, color: SLATE };
    if (score.biased)
      return { label: `biased (${score.cos.toFixed(2)})`, color: ACCENT };
    if (score.aligned)
      return { label: `exact (${score.cos.toFixed(2)})`, color: SAGE };
    return { label: `cos ${score.cos.toFixed(2)}`, color: undefined };
  })();

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
            aria-label={`A gridworld policy. Each cell is shaded by how often it is visited and shows arrows for how likely the policy is to move each way. Expected return is ${pct(score.fractionOfOptimal)} of optimal and the gradient alignment is ${score.cos.toFixed(2)}.`}
            onClick={onCanvasClick}
          >
            {/* Visitation shading: sage where the policy spends its time. */}
            {visitation.map((d, s) => {
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

            {/* The sampled episode path, when running sampled returns. */}
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

        <SimPanel title="What the policy has learned" className="lg:self-start">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">step {state.steps}</Badge>
            <Badge variant="outline">
              return {pct(score.fractionOfOptimal)}
            </Badge>
            <Badge variant="outline">{CRITIC_LABELS[params.critic]}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              variant={verdict.color ? "default" : "outline"}
              style={
                verdict.color
                  ? { backgroundColor: verdict.color, color: "#fff" }
                  : undefined
              }
            >
              gradient {verdict.label}
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

          <ReturnSparkline history={state.rhoHistory} optimal={score.optimal} />

          {inspected !== null ? (
            <div className="text-muted-foreground flex flex-col gap-1 text-xs">
              <span>
                cell ({colOf(inspected)}, {rowOf(inspected)}), visited{" "}
                <span className="text-foreground font-mono">
                  {(visitation[inspected] ?? 0).toFixed(3)}
                </span>
              </span>
              {ACTION_LABELS.map((label, a) => {
                const adv = state.advantage[inspected]?.[a] ?? 0;
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
              Click a cell to read its move probabilities and each move&apos;s
              advantage. The longest arrow is the policy&apos;s favored move;
              clay-orange marks it. Greener cells are visited more often.
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
        <div className="w-40">
          <Slider
            label="Speed"
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
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">critic</span>
          {(
            [
              ["exact", "true value"],
              ["compatible", "compatible"],
              ["incompatible", "incompatible"],
              ["reinforce", "sampled returns"],
            ] as [CriticMode, string][]
          ).map(([mode, label]) => (
            <Toggle
              key={mode}
              label={label}
              active={params.critic === mode}
              onClick={() => intervene({ type: "setCritic", value: mode })}
            />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">value baseline</span>
          <Toggle
            label={params.baseline ? "on" : "off"}
            active={params.baseline}
            onClick={() =>
              intervene({ type: "setBaseline", value: !params.baseline })
            }
          />
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
            label="alpha (learning rate)"
            value={params.alpha}
            min={0.2}
            max={8}
            step={0.1}
            display={params.alpha.toFixed(1)}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          <Slider
            label="gamma (discount)"
            value={params.gamma}
            min={0.8}
            max={0.99}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
          <Slider
            label="episodes per step (sampled returns)"
            value={params.mcEpisodes}
            min={2}
            max={80}
            step={2}
            display={String(params.mcEpisodes)}
            disabled={params.critic !== "reinforce"}
            onChange={(value) => intervene({ type: "setMcEpisodes", value })}
          />
          <Slider
            label="seed (sampled returns)"
            value={state.rng % 1000}
            min={1}
            max={40}
            step={1}
            display={String(state.rng % 1000)}
            disabled={params.critic !== "reinforce"}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </div>
      </SimPanel>

      <p className="text-muted-foreground text-xs">
        The policy is a softmax over a preference per move in each cell, run on
        a 5 by 5 gridworld with a goal worth +1 and a trap worth &minus;1. Every
        step the model solves the policy exactly for its value and its
        discounted visitation, then takes one ascent step using the chosen
        critic. The alignment readout is the cosine between the step actually
        taken and the exact policy gradient, computed separately only to score
        it. It holds at 1.00 for the true value and the compatible critic, drops
        for the incompatible one, and jitters for sampled returns.
      </p>
    </div>
  );
}
