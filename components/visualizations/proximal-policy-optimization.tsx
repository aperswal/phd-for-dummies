"use client";

import { useCallback, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  ACTION_LABELS,
  clipHigh,
  clipLow,
  GOAL,
  initialState,
  PRESETS,
  softmax,
  status,
  step,
  type Intervention,
  type Objective,
  type SimState,
} from "@/components/visualizations/proximal-policy-optimization-model";

const ACCENT = "#c2683f";
const SAGE = "#7c8a5a";

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

const OBJECTIVES: { id: Objective; label: string }[] = [
  { id: "clip", label: "Clip" },
  { id: "cpi", label: "No clip" },
  { id: "kl", label: "KL penalty" },
];

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

export function ClippedSurrogate() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("clip"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("clip");
  const [inspect, setInspect] = useState(0);

  const { params, log } = state;
  const st = status(state);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 320 / speed, () =>
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

  // Layout for the corridor of walkable states (0..GOAL-1) plus the goal tile.
  const width = 1000;
  const margin = 30;
  const cells = GOAL + 1; // walkable states plus the goal
  const slot = (width - margin * 2) / cells;
  const cellY = 60;
  const cellH = 64;
  const cellX = (i: number) => margin + i * slot + 4;
  const cellW = slot - 8;

  const lo = clipLow(params.epsilon);
  const hi = clipHigh(params.epsilon);

  // Ratio band view geometry: map ratio in [0, 2] to a horizontal track.
  const bandTop = 168;
  const bandH = 150;
  const trackX0 = margin + 120;
  const trackX1 = width - margin - 20;
  const ratioToX = (r: number) =>
    trackX0 + (Math.min(Math.max(r, 0), 2) / 2) * (trackX1 - trackX0);

  const inspectProbs = softmax(state.theta, Math.min(inspect, GOAL - 1));
  const inspectRatio = state.ratio[Math.min(inspect, GOAL - 1)] ?? [1, 1];
  const inspectAdv = state.advantageOld[Math.min(inspect, GOAL - 1)] ?? [0, 0];

  // History sparkline geometry.
  const histW = width;
  const histTop = 340;
  const histH = 70;
  const hist = state.rhoHistory;
  const klHist = state.klHistory;
  const histMax = Math.max(st.optimal, 0.1);
  const klMax = 0.6;
  const histPath = (data: number[], scaleMax: number) => {
    if (data.length < 2) return "";
    const n = data.length;
    return data
      .map((v, i) => {
        const x = (i / (n - 1)) * histW;
        const y = histTop + histH - (Math.min(v, scaleMax) / scaleMax) * histH;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

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

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge
          variant="outline"
          style={{
            borderColor: st.inTrustRegion ? SAGE : ACCENT,
            color: st.inTrustRegion ? SAGE : ACCENT,
          }}
        >
          trust region {st.inTrustRegion ? "held" : "broken"}
        </Badge>
        <Badge variant="secondary">round {state.round}</Badge>
        <Badge variant="secondary">
          epoch {state.epoch}/{params.epochs}
        </Badge>
        <Badge variant="outline">KL {st.kl.toFixed(3)}</Badge>
        <Badge variant="outline">
          return {st.rho.toFixed(2)} / {st.optimal.toFixed(2)}
        </Badge>
        <Badge variant="outline">clipped {pct(st.clippedFraction)}</Badge>
      </div>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} 430`}
          className="w-full"
          role="img"
          aria-label={`A corridor of ${GOAL} states. The policy's probability of going right is shown per state, and the probability ratio against the snapshot policy is plotted against the clip band ${lo.toFixed(2)} to ${hi.toFixed(2)}. The trust region is ${st.inTrustRegion ? "held" : "broken"}.`}
        >
          <text
            x={margin}
            y={32}
            className="fill-muted-foreground"
            style={{ fontSize: 13 }}
          >
            the corridor: probability of going right (toward the goal)
          </text>

          {Array.from({ length: cells }).map((_, i) => {
            const isGoal = i === GOAL;
            const probs = isGoal ? [0, 1] : softmax(state.theta, i);
            const right = probs[1] ?? 0;
            const isInspect = i === inspect;
            return (
              <g
                key={`cell-${i}`}
                role="button"
                tabIndex={0}
                aria-label={`Inspect state ${i}`}
                aria-pressed={isInspect}
                className="cursor-pointer outline-none"
                onClick={() => setInspect(i)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setInspect(i);
                  }
                }}
                onMouseEnter={() => setInspect(i)}
              >
                <rect
                  x={cellX(i)}
                  y={cellY}
                  width={cellW}
                  height={cellH}
                  rx={8}
                  fill={isGoal ? SAGE : "var(--color-muted)"}
                  stroke={isInspect ? ACCENT : "transparent"}
                  strokeWidth={2}
                />
                {!isGoal && (
                  <rect
                    x={cellX(i)}
                    y={cellY + cellH - right * cellH}
                    width={cellW}
                    height={right * cellH}
                    rx={8}
                    fill={ACCENT}
                    fillOpacity={0.32}
                  />
                )}
                <text
                  x={cellX(i) + cellW / 2}
                  y={cellY + cellH / 2 + 5}
                  textAnchor="middle"
                  className="fill-foreground"
                  fill={isGoal ? "#fff" : undefined}
                  style={{ fontSize: 14, fontWeight: 600 }}
                >
                  {isGoal ? "goal" : i === 0 ? "start" : `s${i}`}
                </text>
                {!isGoal && (
                  <text
                    x={cellX(i) + cellW / 2}
                    y={cellY + cellH + 16}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 11 }}
                  >
                    {pct(right)}
                  </text>
                )}
              </g>
            );
          })}
          <text
            x={margin}
            y={cellY + cellH + 16}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            left = trap
          </text>

          <text
            x={margin}
            y={bandTop - 12}
            className="fill-muted-foreground"
            style={{ fontSize: 13 }}
          >
            ratio r(theta) vs the clip band, state{" "}
            {inspect <= GOAL - 1 ? inspect : GOAL - 1}
          </text>

          <rect
            x={ratioToX(lo)}
            y={bandTop}
            width={ratioToX(hi) - ratioToX(lo)}
            height={bandH}
            fill={SAGE}
            fillOpacity={0.14}
          />
          <line
            x1={ratioToX(1)}
            x2={ratioToX(1)}
            y1={bandTop}
            y2={bandTop + bandH}
            stroke="var(--color-muted-foreground)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          {[lo, hi].map((edge, k) => (
            <g key={`edge-${k}`}>
              <line
                x1={ratioToX(edge)}
                x2={ratioToX(edge)}
                y1={bandTop}
                y2={bandTop + bandH}
                stroke={SAGE}
                strokeWidth={1.5}
              />
              <text
                x={ratioToX(edge)}
                y={bandTop - 2}
                textAnchor="middle"
                fill={SAGE}
                style={{ fontSize: 10 }}
              >
                {edge.toFixed(2)}
              </text>
            </g>
          ))}
          <text
            x={ratioToX(1)}
            y={bandTop + bandH + 16}
            textAnchor="middle"
            className="fill-muted-foreground"
            style={{ fontSize: 10 }}
          >
            r = 1 (snapshot)
          </text>

          {ACTION_LABELS.map((label, a) => {
            const r = inspectRatio[a] ?? 1;
            const adv = inspectAdv[a] ?? 0;
            const y = bandTop + 38 + a * 56;
            const outside = r > hi || r < lo;
            const dotColor = adv >= 0 ? ACCENT : SAGE;
            return (
              <g key={`ratio-${a}`}>
                <text
                  x={margin}
                  y={y + 4}
                  className="fill-foreground"
                  style={{ fontSize: 12 }}
                >
                  {label} (A {adv >= 0 ? "+" : ""}
                  {adv.toFixed(2)})
                </text>
                <line
                  x1={ratioToX(1)}
                  x2={ratioToX(r)}
                  y1={y}
                  y2={y}
                  stroke={dotColor}
                  strokeWidth={2}
                  strokeOpacity={outside ? 0.4 : 1}
                />
                <circle
                  cx={ratioToX(r)}
                  cy={y}
                  r={7}
                  fill={dotColor}
                  stroke={outside ? ACCENT : "none"}
                  strokeWidth={outside ? 2.5 : 0}
                />
                <text
                  x={ratioToX(r)}
                  y={y - 11}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 10 }}
                >
                  {r.toFixed(2)}
                  {outside ? " clipped" : ""}
                </text>
              </g>
            );
          })}

          <text
            x={margin}
            y={histTop - 8}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            return (orange) and per-round KL (sage) over time
          </text>
          <line
            x1={0}
            x2={histW}
            y1={histTop + histH}
            y2={histTop + histH}
            stroke="var(--color-muted-foreground)"
            strokeOpacity={0.3}
          />
          <path
            d={histPath(hist, histMax)}
            fill="none"
            stroke={ACCENT}
            strokeWidth={2}
          />
          <path
            d={histPath(klHist, klMax)}
            fill="none"
            stroke={SAGE}
            strokeWidth={2}
            strokeOpacity={0.85}
          />
        </svg>
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
            min={0.5}
            max={3}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="The update">
          <div className="flex flex-wrap gap-1.5">
            {OBJECTIVES.map((obj) => (
              <Toggle
                key={obj.id}
                label={obj.label}
                active={params.objective === obj.id}
                onClick={() =>
                  intervene({ type: "setObjective", value: obj.id })
                }
              />
            ))}
          </div>
          <Slider
            label="Clip width epsilon"
            value={params.epsilon}
            min={0.05}
            max={0.8}
            step={0.05}
            display={params.epsilon.toFixed(2)}
            onChange={(value) => intervene({ type: "setEpsilon", value })}
          />
          <Slider
            label="Reuse epochs K (per batch)"
            value={params.epochs}
            min={1}
            max={50}
            step={1}
            display={String(params.epochs)}
            onChange={(value) => intervene({ type: "setEpochs", value })}
          />
          <Slider
            label="Step size"
            value={params.alpha}
            min={0.5}
            max={8}
            step={0.5}
            display={params.alpha.toFixed(1)}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          {params.objective === "kl" && (
            <Slider
              label="KL penalty beta"
              value={params.beta}
              min={0}
              max={8}
              step={0.25}
              display={params.beta.toFixed(2)}
              onChange={(value) => intervene({ type: "setBeta", value })}
            />
          )}
          <Slider
            label="Discount gamma"
            value={params.gamma}
            min={0.8}
            max={0.99}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
          <Slider
            label="Advantage noise"
            value={params.noise}
            min={0}
            max={1}
            step={0.05}
            display={params.noise === 0 ? "off" : params.noise.toFixed(2)}
            onChange={(value) => intervene({ type: "setNoise", value })}
          />
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Minibatch sampling"
              active={params.minibatch}
              onClick={() =>
                intervene({ type: "setMinibatch", value: !params.minibatch })
              }
            />
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => intervene({ type: "resetPolicy" })}
            >
              Reset policy
            </Button>
          </div>
        </SimPanel>

        <SimPanel title={`Inside state ${Math.min(inspect, GOAL - 1)}`}>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">right {pct(inspectProbs[1] ?? 0)}</Badge>
            <Badge variant="outline">
              ratio right {(inspectRatio[1] ?? 1).toFixed(2)}
            </Badge>
            <Badge variant="outline">
              A right {(inspectAdv[1] ?? 0) >= 0 ? "+" : ""}
              {(inspectAdv[1] ?? 0).toFixed(2)}
            </Badge>
          </div>
          <div className="ring-foreground/10 max-h-40 overflow-y-auto rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">action</th>
                  <th className="px-2 py-1 text-right font-medium">prob</th>
                  <th className="px-2 py-1 text-right font-medium">ratio</th>
                  <th className="px-2 py-1 text-right font-medium">A</th>
                  <th className="px-2 py-1 text-right font-medium">clip?</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {ACTION_LABELS.map((label, a) => {
                  const r = inspectRatio[a] ?? 1;
                  const outside = r > hi || r < lo;
                  return (
                    <tr key={`row-${a}`} className="text-muted-foreground">
                      <td className="px-2 py-1">{label}</td>
                      <td className="px-2 py-1 text-right">
                        {pct(inspectProbs[a] ?? 0)}
                      </td>
                      <td className="px-2 py-1 text-right">{r.toFixed(2)}</td>
                      <td className="px-2 py-1 text-right">
                        {(inspectAdv[a] ?? 0).toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {outside ? "yes" : "no"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Play, switch the objective, or drag epsilon to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to round ${event.id}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        Each tick is one inner gradient step on the frozen snapshot batch; after
        K of them the round ends and the snapshot refreshes (Algorithm 1). The
        advantage is solved exactly on this {GOAL}-state corridor, so the
        surrogate is exact up to the softmax policy class; real PPO estimates it
        from samples, which the noise slider stands in for. The trust-region
        light tracks KL(snapshot, policy), the quantity the clip bounds. This
        corridor is forgiving enough that even a divergent no-clip run still
        finds the goal, where on a real task like HalfCheetah the same
        destructive updates sink the return outright.
      </p>
    </div>
  );
}
