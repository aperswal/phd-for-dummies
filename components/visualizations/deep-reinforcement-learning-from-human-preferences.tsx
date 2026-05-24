"use client";

import { useCallback, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  GOAL,
  GRID,
  initialState,
  PRESETS,
  step,
  trueReward,
  type Clip,
  type Intervention,
  type SimState,
} from "@/components/visualizations/deep-reinforcement-learning-from-human-preferences-model";

const ACCENT = "#c2683f";
const SAGE = "#7c8a5a";

// Local typed indexed access so the view does not depend on lib helpers. Under
// the strict compiler every arr[i] is T | undefined; this keeps the hot reads
// typed as T while still failing loudly on a real out-of-range bug.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) throw new RangeError(`index ${index} out of range`);
  return value;
}

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function rowCol(cell: number): { row: number; col: number } {
  return { row: Math.floor(cell / GRID), col: cell % GRID };
}

// Map a learned-reward value in [lo, hi] to a clay-orange fill, dark for high
// reward. Color is paired with the printed number and the policy arrow, so the
// meaning survives without color.
function rewardFill(value: number, lo: number, hi: number): string {
  const span = hi - lo || 1;
  const t = Math.min(Math.max((value - lo) / span, 0), 1);
  const light = 92 - t * 52;
  return `hsl(20 55% ${light}%)`;
}

const ARROWS = ["↑", "↓", "←", "→"];

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
  step,
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
        step={step}
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

function ClipCard({
  clip,
  label,
  onPick,
}: {
  clip: Clip;
  label: string;
  onPick: () => void;
}) {
  const size = 96;
  const cell = size / GRID;
  const trueReturn = clip.cells.reduce((s, c) => s + trueReward(c), 0);
  return (
    <button
      type="button"
      onClick={onPick}
      className="ring-foreground/10 hover:ring-foreground/40 flex flex-col items-center gap-1 rounded-lg p-2 ring-1"
      aria-label={`Prefer clip ${label}, which ends ${
        clip.cells[clip.cells.length - 1] === GOAL
          ? "on the goal"
          : "off the goal"
      }`}
    >
      <span className="text-muted-foreground text-xs font-medium">
        clip {label}
      </span>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        width={size}
        height={size}
        aria-hidden
      >
        {Array.from({ length: GRID * GRID }, (_, c) => {
          const { row, col } = rowCol(c);
          return (
            <rect
              key={`bg-${c}`}
              x={col * cell}
              y={row * cell}
              width={cell - 1}
              height={cell - 1}
              rx={1.5}
              fill={c === GOAL ? SAGE : "var(--color-muted)"}
              fillOpacity={c === GOAL ? 0.7 : 0.4}
            />
          );
        })}
        {clip.cells.map((c, i) => {
          if (i === 0) return null;
          const from = rowCol(at(clip.cells, i - 1));
          const to = rowCol(c);
          return (
            <line
              key={`seg-${i}`}
              x1={from.col * cell + cell / 2}
              y1={from.row * cell + cell / 2}
              x2={to.col * cell + cell / 2}
              y2={to.row * cell + cell / 2}
              stroke={ACCENT}
              strokeWidth={2.5}
              strokeLinecap="round"
            />
          );
        })}
        {clip.cells.map((c, i) => {
          const { row, col } = rowCol(c);
          return (
            <circle
              key={`dot-${i}`}
              cx={col * cell + cell / 2}
              cy={row * cell + cell / 2}
              r={i === clip.cells.length - 1 ? 4 : 2.5}
              fill={ACCENT}
            />
          );
        })}
      </svg>
      <span className="text-muted-foreground font-mono text-[10px]">
        true return {trueReturn.toFixed(2)}
      </span>
    </button>
  );
}

export function PreferenceLoop() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("oracle"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("oracle");

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const waitingForLabel =
    state.params.labeler === "you" && state.pending !== null;

  const resetClock = useFixedTimestep(
    playing && !waitingForLabel,
    520 / speed,
    () => dispatch({ kind: "tick" }),
  );

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    dispatch({ kind: "reset", presetId });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  const { params, rhat, policy, agentCell, pending } = state;
  const lo = Math.min(...rhat);
  const hi = Math.max(...rhat);

  const size = 360;
  const gap = 4;
  const cellSize = (size - gap * (GRID + 1)) / GRID;
  const at2 = (i: number) => gap + i * (cellSize + gap);

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

      <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
        <div className="flex flex-col items-center gap-2">
          <span className="text-muted-foreground text-xs font-medium">
            learned reward and its greedy policy
          </span>
          <svg
            viewBox={`0 0 ${size} ${size}`}
            width={size}
            height={size}
            className="ring-foreground/10 rounded-xl ring-1"
            role="img"
            aria-label={`A ${GRID} by ${GRID} grid colored by the learned reward, with arrows showing the greedy policy. The agent sits at row ${
              rowCol(agentCell).row + 1
            }, column ${rowCol(agentCell).col + 1}.`}
          >
            <rect width={size} height={size} fill="var(--color-card)" />
            {Array.from({ length: GRID * GRID }, (_, c) => {
              const { row, col } = rowCol(c);
              const value = at(rhat, c);
              const isGoal = c === GOAL;
              return (
                <g key={`cell-${c}`}>
                  <rect
                    x={at2(col)}
                    y={at2(row)}
                    width={cellSize}
                    height={cellSize}
                    rx={6}
                    fill={isGoal ? SAGE : rewardFill(value, lo, hi)}
                    stroke={isGoal ? SAGE : "transparent"}
                    strokeWidth={2}
                  />
                  {!isGoal && (
                    <text
                      x={at2(col) + cellSize / 2}
                      y={at2(row) + cellSize / 2 - 4}
                      textAnchor="middle"
                      className="fill-foreground/70"
                      style={{ fontSize: 11 }}
                    >
                      {ARROWS[at(policy, c)]}
                    </text>
                  )}
                  <text
                    x={at2(col) + cellSize / 2}
                    y={at2(row) + cellSize / 2 + 12}
                    textAnchor="middle"
                    fill={isGoal ? "#fff" : "var(--color-foreground)"}
                    fillOpacity={isGoal ? 1 : 0.55}
                    className="font-mono"
                    style={{ fontSize: 9 }}
                  >
                    {isGoal ? "goal" : value.toFixed(2)}
                  </text>
                </g>
              );
            })}
            <circle
              cx={at2(rowCol(agentCell).col) + cellSize / 2}
              cy={at2(rowCol(agentCell).row) + cellSize / 2}
              r={cellSize * 0.22}
              fill={ACCENT}
              stroke="#fff"
              strokeWidth={2}
            />
          </svg>
          <p className="text-muted-foreground max-w-[360px] text-center text-[11px]">
            The agent never sees the true reward. It only chases this learned
            map, fit from clip comparisons. The green cell is the hidden goal.
          </p>
        </div>

        <SimPanel
          title={waitingForLabel ? "Which clip is better?" : "Reward fitting"}
        >
          {waitingForLabel && pending ? (
            <div className="flex flex-col gap-2">
              <p className="text-muted-foreground text-xs">
                Pick the clip whose path ends nearer the green goal. Tie if they
                look equal.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-3">
                <ClipCard
                  clip={pending.a}
                  label="A"
                  onPick={() => intervene({ type: "label", prefer: "a" })}
                />
                <ClipCard
                  clip={pending.b}
                  label="B"
                  onPick={() => intervene({ type: "label", prefer: "b" })}
                />
              </div>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => intervene({ type: "label", prefer: "tie" })}
              >
                Can&apos;t tell, call it a tie
              </Button>
            </div>
          ) : (
            <div className="text-muted-foreground flex flex-col gap-2 text-xs">
              <p>
                Labeled comparisons train the reward map. The oracle labels
                automatically; switch to &quot;You are the labeler&quot; to pick
                clips yourself.
              </p>
              <p>
                Labels so far:{" "}
                <span className="text-foreground font-mono">
                  {state.labelCount}
                </span>
                . Database holds the most recent{" "}
                <span className="text-foreground font-mono">
                  {state.database.length}
                </span>
                .
              </p>
            </div>
          )}
        </SimPanel>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={waitingForLabel && !playing}
        />
        <div className="w-40">
          <Slider
            label="Speed"
            value={speed}
            min={0.5}
            max={3}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Oracle labels"
              active={params.labeler === "oracle"}
              onClick={() => intervene({ type: "setLabeler", value: "oracle" })}
            />
            <Toggle
              label="You label"
              active={params.labeler === "you"}
              onClick={() => intervene({ type: "setLabeler", value: "you" })}
            />
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Query by disagreement"
              active={params.queryRule === "disagreement"}
              onClick={() =>
                intervene({ type: "setQueryRule", value: "disagreement" })
              }
            />
            <Toggle
              label="Random queries"
              active={params.queryRule === "random"}
              onClick={() =>
                intervene({ type: "setQueryRule", value: "random" })
              }
            />
          </div>
          <Toggle
            label={
              params.onlineQueries
                ? "Online queries: on"
                : "Online queries: off"
            }
            active={params.onlineQueries}
            onClick={() => intervene({ type: "toggleOnline" })}
          />
          <Slider
            label="Mislabel rate (human error)"
            value={params.mislabelRate}
            min={0}
            max={0.5}
            step={0.05}
            display={`${Math.round(params.mislabelRate * 100)}%`}
            onChange={(value) => intervene({ type: "setMislabel", value })}
          />
          <Slider
            label="Query every N steps (lower = more feedback)"
            value={params.queryEvery}
            min={1}
            max={12}
            step={1}
            display={`${params.queryEvery}`}
            onChange={(value) => intervene({ type: "setQueryEvery", value })}
          />
        </SimPanel>

        <SimPanel title="How aligned is the learned reward?">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              variant={state.alignment > 0.6 ? "default" : "outline"}
              style={
                state.alignment > 0.6 ? { backgroundColor: SAGE } : undefined
              }
            >
              reward match {(state.alignment * 100).toFixed(0)}%
            </Badge>
            <Badge
              variant={state.optimalShare > 0.7 ? "default" : "outline"}
              style={
                state.optimalShare > 0.7 ? { backgroundColor: SAGE } : undefined
              }
            >
              optimal policy {(state.optimalShare * 100).toFixed(0)}%
            </Badge>
            <Badge variant="outline">BT loss {state.loss.toFixed(3)}</Badge>
            <Badge variant="secondary">step {state.step}</Badge>
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-[11px]">
              reward match: how closely the learned map tracks the hidden true
              reward
            </span>
            <div className="bg-foreground/10 h-2 w-full overflow-hidden rounded">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.max(0, state.alignment) * 100}%`,
                  backgroundColor: ACCENT,
                }}
              />
            </div>
            <span className="text-muted-foreground text-[11px]">
              optimal policy: share of cells whose arrow matches the true
              optimum
            </span>
            <div className="bg-foreground/10 h-2 w-full overflow-hidden rounded">
              <div
                className="h-full rounded"
                style={{
                  width: `${state.optimalShare * 100}%`,
                  backgroundColor: SAGE,
                }}
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Press play, or label a clip, to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.snapshot.step}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The reward fitting here is the paper&apos;s real Bradley-Terry loss over
        clip pairs, trained online by gradient descent, with an ensemble of{" "}
        {state.predictors.length} predictors estimating disagreement for query
        selection. The world is a {GRID} by {GRID} grid where the learner never
        sees the true reward, standing in for the paper&apos;s Atari and MuJoCo
        tasks; a deep network there learns a reward over pixels, here it is one
        number per cell so the whole loop fits on screen.
      </p>
    </div>
  );
}
