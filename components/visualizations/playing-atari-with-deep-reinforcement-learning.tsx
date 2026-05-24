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
const ACCENT_SOFT = "#e0b59f";
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
}

function Slider({
  label,
  value,
  min,
  max,
  step: stepSize,
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
        step={stepSize}
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

// A small sparkline of recent maxAbsQ. The y axis is log-ish so a calm run reads
// flat near the bottom and a divergent run shoots to the top.
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

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 110 / speed, () =>
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

  // SVG layout for the catch grid plus the per-action value bars.
  const width = 1000;
  const cell = 56;
  const gridX = 28;
  const gridY = 24;
  const gridPx = GRID_W * cell;
  const barsX = gridX + gridPx + 70;
  const barMax = 200;
  const barW = 64;
  const cx = (col: number) => gridX + col * cell + cell / 2;

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

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} 330`}
          className="w-full"
          role="img"
          aria-label={`A catch game grid with the ball at column ${game.ballCol}, row ${game.ballRow}, and the paddle at column ${game.paddleCol}. The Q-network favours the action '${ACTIONS[bestAction]}'.`}
        >
          {Array.from({ length: GRID_W }).map((_, col) =>
            Array.from({ length: 5 }).map((__, row) => (
              <rect
                key={`g-${col}-${row}`}
                x={gridX + col * cell + 2}
                y={gridY + row * cell + 2}
                width={cell - 4}
                height={cell - 4}
                rx={6}
                className="fill-foreground/5"
              />
            )),
          )}

          <circle
            cx={cx(game.ballCol)}
            cy={gridY + game.ballRow * cell + cell / 2}
            r={cell / 2 - 9}
            fill={ACCENT}
          />

          <rect
            x={gridX + game.paddleCol * cell + 6}
            y={gridY + 4 * cell + cell / 2 - 7}
            width={cell - 12}
            height={14}
            rx={5}
            fill={SAGE}
          />

          <text
            x={gridX}
            y={gridY + 5 * cell + 22}
            className="fill-muted-foreground"
            style={{ fontSize: 13 }}
          >
            ball drops, paddle catches
          </text>

          {ACTIONS.map((name, a) => {
            const value = q[a] ?? 0;
            const norm = Math.max(-1, Math.min(1, value / 2));
            const magnitude = Math.abs(norm) * (barMax / 2);
            const zero = gridY + barMax / 2;
            const x = barsX + a * (barW + 26);
            const top = norm >= 0 ? zero - magnitude : zero;
            const isBest = a === bestAction;
            return (
              <g key={`bar-${name}`}>
                <rect
                  x={x}
                  y={top}
                  width={barW}
                  height={Math.max(magnitude, 1)}
                  rx={4}
                  fill={isBest ? ACCENT : ACCENT_SOFT}
                />
                <text
                  x={x + barW / 2}
                  y={zero + barMax / 2 + 20}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{ fontSize: 13, fontWeight: isBest ? 600 : 400 }}
                >
                  {name}
                </text>
                <text
                  x={x + barW / 2}
                  y={norm >= 0 ? top - 6 : top + magnitude + 14}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 11 }}
                >
                  {value.toFixed(2)}
                </text>
              </g>
            );
          })}
          <line
            x1={barsX - 14}
            y1={gridY + barMax / 2}
            x2={barsX + 3 * (barW + 26)}
            y2={gridY + barMax / 2}
            stroke="currentColor"
            strokeOpacity={0.2}
          />
          <text
            x={barsX}
            y={gridY - 4}
            className="fill-muted-foreground"
            style={{ fontSize: 13 }}
          >
            Q-value per action
          </text>
        </svg>
      </div>

      <div className="flex flex-col gap-1">
        <div className="text-muted-foreground flex items-center justify-between text-xs">
          <span>value estimate over time (log scale)</span>
          <span
            className={
              metrics.diverged
                ? "font-medium text-[color:var(--accent-orange,#c2683f)]"
                : ""
            }
            style={metrics.diverged ? { color: ACCENT } : undefined}
          >
            {metrics.diverged ? "value diverged" : "value bounded"}
          </span>
        </div>
        <div className="text-foreground ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <QPlot history={metrics.qHistory} diverged={metrics.diverged} />
        </div>
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
            max={10}
            step={1}
            display={`${speed}x`}
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
              label="Pin ball (shift)"
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
            label="Replay buffer size"
            value={params.bufferCapacity}
            min={2}
            max={800}
            step={2}
            display={String(params.bufferCapacity)}
            onChange={(value) =>
              intervene({ type: "setBufferCapacity", value })
            }
          />
          <Slider
            label="Minibatch size"
            value={params.batchSize}
            min={1}
            max={32}
            step={1}
            display={String(params.batchSize)}
            onChange={(value) => intervene({ type: "setBatchSize", value })}
          />
          <Slider
            label="Learning rate"
            value={params.learningRate}
            min={0.02}
            max={0.5}
            step={0.01}
            display={params.learningRate.toFixed(2)}
            onChange={(value) => intervene({ type: "setLearningRate", value })}
          />
          <Slider
            label="Epsilon (exploration)"
            value={params.epsilon}
            min={0}
            max={1}
            step={0.01}
            display={params.epsilon.toFixed(2)}
            onChange={(value) => intervene({ type: "setEpsilon", value })}
          />
          <Slider
            label="Gamma (discount)"
            value={params.gamma}
            min={0.5}
            max={0.99}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the loop">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">step {metrics.step}</Badge>
            <Badge variant={params.replayOn ? "outline" : "default"}>
              replay {params.replayOn ? "on" : "off"}
            </Badge>
            <Badge variant="outline">best move {ACTIONS[bestAction]}</Badge>
            <Badge variant={metrics.diverged ? "default" : "outline"}>
              max |Q|{" "}
              {metrics.maxAbsQ < 1000 ? metrics.maxAbsQ.toFixed(2) : "blown up"}
            </Badge>
            <Badge variant="outline">avg return {avgReturn.toFixed(2)}</Badge>
          </div>
          <div className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1">
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
                      {a === bestAction ? "greedy" : ""}
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
        toward the target r + gamma&middot;max Q(s&apos;). Two honest
        simplifications stand in for the paper&apos;s scale. The game is a 5 by
        5 catch grid, not an Atari screen, and the Q-network is a linear
        approximator over a few overlapping tile features, not a convolutional
        net. The overlap is on purpose, since it recreates the deadly triad the
        paper warns about. With replay off the value estimate runs away; with
        replay on the same learning rate stays bounded.
      </p>
    </div>
  );
}
