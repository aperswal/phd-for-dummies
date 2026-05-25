"use client";

import { useCallback, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  ACTIONS,
  featurize,
  GRID_W,
  initialState,
  meanReturn,
  PRESETS,
  qValues,
  step,
  type Input,
  type Intervention,
  type SimState,
} from "@/components/visualizations/playing-atari-with-deep-reinforcement-learning-model";

const ACCENT = "#c2683f";
const SAGE = "#7d8a6a";

function reducer(state: SimState, action: Input): SimState {
  return step(state, action);
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
  ariaLabel: string;
}

function Slider({
  label,
  value,
  min,
  max,
  step: stepSize,
  display,
  onChange,
  ariaLabel,
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
        step={stepSize}
        value={value}
        aria-label={ariaLabel}
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

// A small sparkline of recent maxAbsQ. The y axis is log-ish so a calm run
// reads flat near the bottom and a divergent run shoots to the top.
function QPlot({
  history,
  diverged,
}: {
  history: number[];
  diverged: boolean;
}) {
  const w = 1000;
  const h = 90;
  const cap = Math.max(2, ...history);
  const scaled = (v: number) =>
    h - (Math.log10(1 + v) / Math.log10(1 + cap)) * h;
  const points = history
    .map((v, i) => `${(i / Math.max(history.length - 1, 1)) * w},${scaled(v)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" aria-hidden>
      <line
        x1={0}
        y1={h - 1}
        x2={w}
        y2={h - 1}
        stroke="currentColor"
        strokeOpacity={0.15}
      />
      {history.length > 1 && (
        <polyline
          points={points}
          fill="none"
          stroke={diverged ? ACCENT : SAGE}
          strokeWidth={diverged ? 4 : 2.5}
        />
      )}
    </svg>
  );
}

// The catch-game grid rendered as a standalone SVG so it can be sized
// independently of the Q-value display.
function GameGrid({
  ballCol,
  ballRow,
  paddleCol,
  bestActionLabel,
}: {
  ballCol: number;
  ballRow: number;
  paddleCol: number;
  bestActionLabel: string;
}) {
  // Cell is sized so the 5-column grid fills a ~300px canvas comfortably.
  const cell = 52;
  const gridX = 20;
  const gridY = 20;
  const viewW = gridX * 2 + GRID_W * cell;
  const viewH = gridY + 5 * cell + 32;
  const cx = (col: number) => gridX + col * cell + cell / 2;

  return (
    <svg
      viewBox={`0 0 ${viewW} ${viewH}`}
      className="w-full"
      role="img"
      aria-label={`Catch game: ball at column ${ballCol}, row ${ballRow}; paddle at column ${paddleCol}. Network prefers action: ${bestActionLabel}.`}
    >
      {Array.from({ length: GRID_W }).map((_, col) =>
        Array.from({ length: 5 }).map((__, row) => (
          <rect
            key={`g-${col}-${row}`}
            x={gridX + col * cell + 2}
            y={gridY + row * cell + 2}
            width={cell - 4}
            height={cell - 4}
            rx={5}
            className="fill-foreground/5"
          />
        )),
      )}

      <circle
        cx={cx(ballCol)}
        cy={gridY + ballRow * cell + cell / 2}
        r={cell / 2 - 8}
        fill={ACCENT}
      />

      <rect
        x={gridX + paddleCol * cell + 5}
        y={gridY + 4 * cell + cell / 2 - 6}
        width={cell - 10}
        height={12}
        rx={4}
        fill={SAGE}
      />

      <text
        x={gridX}
        y={gridY + 5 * cell + 20}
        className="fill-muted-foreground"
        style={{ fontSize: 12 }}
      >
        ball drops — paddle catches
      </text>
    </svg>
  );
}

// Q-value bar chart rendered in HTML so it scales naturally at any width.
function QValueBars({
  q,
  bestAction,
}: {
  q: number[];
  bestAction: number;
}) {
  // Normalize bars against the max absolute Q so the chart always uses its
  // full height, and label the chosen action clearly.
  const maxAbs = Math.max(1, ...q.map(Math.abs));
  return (
    <div className="flex flex-col gap-1 p-4">
      <span className="text-muted-foreground mb-1 text-xs">
        Q-value per action
      </span>
      <div className="flex items-end gap-2" style={{ height: 96 }}>
        {ACTIONS.map((name, a) => {
          const value = q[a] ?? 0;
          const pct = Math.abs(value) / maxAbs;
          const isBest = a === bestAction;
          return (
            <div
              key={`qbar-${name}`}
              className="flex flex-1 flex-col items-center gap-1"
            >
              {/* value label above bar */}
              <span
                className="text-muted-foreground font-mono text-xs"
                aria-hidden
              >
                {value.toFixed(2)}
              </span>
              {/* bar */}
              <div
                className="w-full rounded"
                style={{
                  height: `${Math.max(pct * 72, 2)}px`,
                  backgroundColor: isBest ? ACCENT : "currentColor",
                  opacity: isBest ? 1 : 0.18,
                }}
              />
              {/* action name */}
              <span
                className="text-xs"
                style={{
                  fontWeight: isBest ? 600 : 400,
                  color: isBest ? "inherit" : undefined,
                }}
              >
                {name}
                {isBest ? " ★" : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function AtariReplay() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("stable"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [presetId, setPresetId] = useState("stable");

  const { params, game, metrics, log } = state;
  const features = featurize(game);
  const q = qValues(state.weights, features);
  const bestAction = q.indexOf(Math.max(...q));
  const avgReturn = meanReturn(metrics.returnsWindow);

  // When the run diverges the model freezes. Derive an effective playing flag
  // that stops the fixed-timestep timer automatically without setState-in-effect.
  // The Play button reflects this derived state so it reads "paused" on divergence.
  const isPlaying = playing && !metrics.diverged;

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(isPlaying, 110 / speed, () =>
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
    intervene({ type: "loadPreset", id: presetId });
  }, [intervene, presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      intervene({ type: "loadPreset", id });
    },
    [intervene, resetClock],
  );

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      {/* Spine: preset scenarios that teach the paper's core contrast */}
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

      {/* Game scene: grid and Q-values stacked so both are legible at any width */}
      <div className="ring-foreground/10 grid gap-0 overflow-hidden rounded-xl ring-1 sm:grid-cols-2">
        <div className="border-foreground/10 p-4 sm:border-r">
          <GameGrid
            ballCol={game.ballCol}
            ballRow={game.ballRow}
            paddleCol={game.paddleCol}
            bestActionLabel={ACTIONS[bestAction] ?? "stay"}
          />
        </div>
        <QValueBars q={q} bestAction={bestAction} />
      </div>

      {/* Value estimate sparkline */}
      <div className="flex flex-col gap-1">
        <div className="text-muted-foreground flex items-center justify-between text-xs">
          <span>value estimate over time (log scale)</span>
          {metrics.diverged && (
            <span
              className="font-semibold"
              style={{ color: ACCENT }}
              role="status"
              aria-live="polite"
            >
              ⚠ value diverged — replay stopped
            </span>
          )}
          {!metrics.diverged && (
            <span className="text-muted-foreground">value bounded</span>
          )}
        </div>
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <QPlot history={metrics.qHistory} diverged={metrics.diverged} />
        </div>
      </div>

      {/* Transport controls + playback speed (pace, not paper mechanism) */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={isPlaying}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="w-40">
          <Slider
            label="Playback speed"
            value={speed}
            min={1}
            max={10}
            step={1}
            display={`${speed}×`}
            ariaLabel="Playback speed — controls how fast the animation plays, not anything in the paper"
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Experience replay"
              active={params.replayOn}
              onClick={() => intervene({ type: "toggleReplay" })}
            />
            <Toggle
              label="Pin ball to one column"
              active={params.forceColumn >= 0}
              onClick={() =>
                intervene({
                  type: "setForceColumn",
                  value: params.forceColumn >= 0 ? -1 : 0,
                })
              }
            />
          </div>
          <Slider
            label="Replay buffer size — how many past transitions the memory holds"
            value={params.bufferCapacity}
            min={2}
            max={800}
            step={2}
            display={String(params.bufferCapacity)}
            ariaLabel="Replay buffer capacity"
            onChange={(value) =>
              intervene({ type: "setBufferCapacity", value })
            }
          />
          <Slider
            label="Minibatch size — transitions sampled per update step"
            value={params.batchSize}
            min={1}
            max={32}
            step={1}
            display={String(params.batchSize)}
            ariaLabel="Minibatch size"
            onChange={(value) => intervene({ type: "setBatchSize", value })}
          />
          <Slider
            label="Learning rate — step size for each weight update"
            value={params.learningRate}
            min={0.02}
            max={0.5}
            step={0.01}
            display={params.learningRate.toFixed(2)}
            ariaLabel="Learning rate"
            onChange={(value) => intervene({ type: "setLearningRate", value })}
          />
          <Slider
            label="Epsilon — fraction of steps the agent picks a random action"
            value={params.epsilon}
            min={0}
            max={1}
            step={0.01}
            display={params.epsilon.toFixed(2)}
            ariaLabel="Epsilon exploration rate"
            onChange={(value) => intervene({ type: "setEpsilon", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the loop">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">step {metrics.step}</Badge>
            <Badge variant={params.replayOn ? "outline" : "secondary"}>
              replay {params.replayOn ? "on" : "off"}
            </Badge>
            <Badge variant="outline">
              best move {ACTIONS[bestAction] ?? "stay"}
            </Badge>
            <Badge variant={metrics.diverged ? "destructive" : "outline"}>
              max |Q|{" "}
              {metrics.maxAbsQ < 1000
                ? metrics.maxAbsQ.toFixed(2)
                : "blown up"}
            </Badge>
            <Badge variant="outline">avg return {avgReturn.toFixed(2)}</Badge>
          </div>
          <div className="ring-foreground/10 rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">action</th>
                  <th className="px-2 py-1 text-right font-medium">Q-value</th>
                  <th className="px-2 py-1 text-right font-medium">chosen</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {ACTIONS.map((name, a) => (
                  <tr
                    key={`row-${name}`}
                    className={
                      a === bestAction
                        ? "text-foreground"
                        : "text-muted-foreground"
                    }
                  >
                    <td className="px-2 py-1">{name}</td>
                    <td className="px-2 py-1 text-right">
                      {(q[a] ?? 0).toFixed(3)}
                    </td>
                    <td className="px-2 py-1 text-right">
                      {a === bestAction ? "greedy ★" : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Flip replay, drag a slider, or pin the ball to start the log."
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
        This runs the inner loop of Algorithm 1 exactly: behave
        epsilon-greedily, store each transition, sample a minibatch, and regress
        toward the target r + &gamma;&middot;max Q(s&apos;). Two honest
        simplifications stand in for the paper&apos;s scale. The game is a 5
        &times; 5 catch grid, not an Atari screen, and the Q-network is a linear
        approximator over a few overlapping tile features, not a convolutional
        net. The overlap is on purpose — it recreates the deadly triad the paper
        warns about. With replay off the value estimate runs away; with replay on
        the same learning rate stays bounded. The discount factor gamma is fixed
        at 0.95; the run is deterministic for any given preset.
      </p>
    </div>
  );
}
