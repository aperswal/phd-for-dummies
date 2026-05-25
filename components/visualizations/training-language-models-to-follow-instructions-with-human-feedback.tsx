"use client";

import { useCallback, useMemo, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  BASE_STYLES,
  initialState,
  PRESETS,
  readout,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/training-language-models-to-follow-instructions-with-human-feedback-model";

const ACCENT = "#c2683f";
const SAGE = "#7d8a6a";
const ACCENT_SOFT = "#e0b59f";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  if (action.kind === "tick") return step(state, { kind: "tick" });
  return step(state, { kind: "intervene", action: action.action });
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
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
  step,
  display,
  onChange,
}: SliderProps) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-muted-foreground flex items-start justify-between gap-2">
        <span className="leading-snug">{label}</span>
        <span className="text-foreground shrink-0 font-mono">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer"
        aria-label={label}
        style={{ accentColor: ACCENT }}
      />
    </label>
  );
}

interface Trace {
  step: number;
  proxy: number;
  trueQuality: number;
  capability: number;
}

const PLOT_MAX = 90;

function traceFrom(state: SimState): Trace[] {
  const r = readout(state);
  return [
    {
      step: state.step,
      proxy: r.proxyReward,
      trueQuality: r.trueQuality,
      capability: r.capability,
    },
  ];
}

export function TrainingWithHumanFeedback() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("aligned"),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState("aligned");
  const [inspect, setInspect] = useState<number>(0);

  // The trajectory the plot draws, held in state so the chart repaints. A ref
  // mirrors the live model so a callback can read the freshest value and step it
  // forward without reading or writing state during render. Every mutation goes
  // through these callbacks, never through render, so the ref and the reducer
  // (both pure functions of the same inputs) stay in lockstep.
  const stateRef = useRef(state);
  const [trace, setTrace] = useState<Trace[]>(() => traceFrom(state));

  const view = useMemo(() => readout(state), [state]);
  const styles = BASE_STYLES;

  const intervene = useCallback((action: Intervention) => {
    stateRef.current = step(stateRef.current, { kind: "intervene", action });
    dispatch({ kind: "intervene", action });
  }, []);

  // One model update plus one point appended to the trajectory, both derived
  // from the freshest state in the ref so the chart and the model never drift.
  const doTick = useCallback(() => {
    const next = step(stateRef.current, { kind: "tick" });
    stateRef.current = next;
    dispatch({ kind: "tick" });
    const r = readout(next);
    setTrace((prev) => {
      const point: Trace = {
        step: next.step,
        proxy: r.proxyReward,
        trueQuality: r.trueQuality,
        capability: r.capability,
      };
      const appended = [...prev, point];
      return appended.length > PLOT_MAX ? appended.slice(-PLOT_MAX) : appended;
    });
  }, []);

  const resetClock = useFixedTimestep(playing, 420, doTick);

  const reset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      const fresh = initialState(id);
      stateRef.current = fresh;
      setTrace(traceFrom(fresh));
      dispatch({ kind: "reset", presetId: id });
    },
    [resetClock],
  );

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onReset = useCallback(() => reset(presetId), [presetId, reset]);
  const loadPreset = useCallback(
    (id: string) => {
      setPresetId(id);
      reset(id);
    },
    [reset],
  );

  const argmax = view.pi.indexOf(Math.max(...view.pi));

  // Layout for the policy-bar panel.
  const width = 1000;
  const barAreaH = 200;
  const barTop = 28;
  const barBottom = barTop + barAreaH;
  const slot = (width - 56) / styles.length;
  const barX = (i: number) => 28 + i * slot + 6;
  const barW = slot - 12;

  // Layout for the trajectory plot.
  const plotW = width;
  const plotH = 180;
  const plotPad = 34;
  const plotInnerW = plotW - plotPad * 2;
  const plotInnerH = plotH - plotPad * 2;
  const yMax = 1.25;
  const px = (i: number) =>
    plotPad + (trace.length <= 1 ? 0 : (i / (PLOT_MAX - 1)) * plotInnerW);
  const py = (value: number) =>
    plotPad + plotInnerH - (Math.min(value, yMax) / yMax) * plotInnerH;

  const linePath = (key: keyof Trace) =>
    trace
      .map((t, i) => `${i === 0 ? "M" : "L"} ${px(i)} ${py(t[key] as number)}`)
      .join(" ");

  const current = view;
  const hacking = current.proxyReward - current.trueQuality;

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
          viewBox={`0 0 ${width} ${barBottom + 60}`}
          className="w-full"
          role="img"
          aria-label={`The policy's chance of producing each response style. The most likely style is ${styles[argmax]?.name} at ${percent(current.pi[argmax] ?? 0)}.`}
        >
          {[0.25, 0.5, 0.75, 1].map((g) => (
            <line
              key={`grid-${g}`}
              x1={28}
              x2={width - 28}
              y1={barBottom - g * barAreaH}
              y2={barBottom - g * barAreaH}
              className="stroke-foreground/10"
              strokeWidth={1}
            />
          ))}
          {styles.map((style, i) => {
            const p = current.pi[i] ?? 0;
            const rm = current.rmScores[i] ?? 0;
            const h = p * barAreaH;
            const x = barX(i);
            const isTop = i === argmax;
            const isInspect = i === inspect;
            return (
              <g
                key={style.name}
                role="button"
                tabIndex={0}
                aria-label={`Inspect and nudge the ${style.name} style`}
                aria-pressed={isInspect}
                className="cursor-pointer outline-none"
                onClick={() => setInspect(i)}
                onFocus={() => setInspect(i)}
                onMouseEnter={() => setInspect(i)}
              >
                <rect
                  x={x}
                  y={barTop}
                  width={barW}
                  height={barAreaH}
                  rx={4}
                  className="fill-foreground/5"
                  stroke={isInspect ? ACCENT : "transparent"}
                  strokeWidth={1.5}
                  strokeDasharray={isInspect ? "4 3" : undefined}
                />
                <rect
                  x={x}
                  y={barBottom - h}
                  width={barW}
                  height={h}
                  rx={4}
                  fill={isTop ? ACCENT : ACCENT_SOFT}
                />
                {/* reward-model score: a thin sage tick above the policy bar */}
                <line
                  x1={x}
                  x2={x + barW}
                  y1={barBottom - rm * barAreaH}
                  y2={barBottom - rm * barAreaH}
                  stroke={SAGE}
                  strokeWidth={3}
                  strokeLinecap="round"
                />
                {p > 0.03 && (
                  <text
                    x={x + barW / 2}
                    y={barBottom - h - 6}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 11 }}
                  >
                    {percent(p)}
                  </text>
                )}
                <text
                  x={x + barW / 2}
                  y={barBottom + 18}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{
                    fontSize: Math.min(13, slot * 0.18),
                    fontWeight: 500,
                  }}
                >
                  {style.name}
                </text>
              </g>
            );
          })}
          <text
            x={28}
            y={barTop - 10}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            policy: chance of each response style (bar) vs reward-model score
            (sage tick)
          </text>
          <text
            x={28}
            y={barBottom + 44}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            click a bar to inspect and nudge that style
          </text>
        </svg>
      </div>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${plotW} ${plotH}`}
          className="w-full"
          role="img"
          aria-label={`Proxy reward versus true quality over training. Proxy reward is ${current.proxyReward.toFixed(2)} and true quality is ${current.trueQuality.toFixed(2)}.`}
        >
          {[0.5, 1].map((g) => (
            <g key={`pg-${g}`}>
              <line
                x1={plotPad}
                x2={plotW - plotPad}
                y1={py(g)}
                y2={py(g)}
                className="stroke-foreground/10"
              />
              <text
                x={plotPad - 6}
                y={py(g) + 3}
                textAnchor="end"
                className="fill-muted-foreground"
                style={{ fontSize: 10 }}
              >
                {g.toFixed(1)}
              </text>
            </g>
          ))}
          {trace.length > 1 && (
            <>
              <path
                d={linePath("proxy")}
                fill="none"
                stroke={SAGE}
                strokeWidth={2.5}
              />
              <path
                d={linePath("trueQuality")}
                fill="none"
                stroke={ACCENT}
                strokeWidth={2.5}
              />
              <path
                d={linePath("capability")}
                fill="none"
                stroke="var(--color-muted-foreground)"
                strokeWidth={1.5}
                strokeDasharray="4 4"
              />
            </>
          )}
          <text
            x={plotPad}
            y={16}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            <tspan fill={SAGE}>proxy reward</tspan> vs{" "}
            <tspan fill={ACCENT}>true quality</tspan> vs capability (dashed)
            over updates
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={doTick}
          onReset={onReset}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="The RL objective (equation 2)">
          <Slider
            label="KL penalty beta — how tightly the leash holds the policy near the SFT starting point (drag to 0 on Reward hacking to watch true quality collapse)"
            value={state.params.beta}
            min={0}
            max={0.6}
            step={0.01}
            display={state.params.beta.toFixed(2)}
            onChange={(value) => intervene({ type: "setBeta", value })}
          />
          <Slider
            label="Pretraining mix gamma — how much the original pretraining objective is mixed back in to recover lost capability (0 = plain PPO; raise on Alignment tax preset)"
            value={state.params.gamma}
            min={0}
            max={0.6}
            step={0.05}
            display={state.params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs">
              Corrupt the reward model on{" "}
              <span className="text-foreground font-medium">
                {styles[inspect]?.name}
              </span>{" "}
              — over-rate it and watch the policy chase it; undo with under-rate
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() =>
                  intervene({
                    type: "corruptReward",
                    index: inspect,
                    amount: 0.2,
                  })
                }
              >
                over-rate +0.2
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() =>
                  intervene({
                    type: "corruptReward",
                    index: inspect,
                    amount: -0.2,
                  })
                }
              >
                under-rate −0.2
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() =>
                  intervene({ type: "nudgeStyle", index: inspect, amount: 0.8 })
                }
              >
                nudge policy +
              </Button>
            </div>
          </div>
        </SimPanel>

        <SimPanel title={`Inside the loop (step ${state.step})`}>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              style={{
                backgroundColor: SAGE,
                color: "var(--color-background, #fff)",
              }}
            >
              proxy reward {current.proxyReward.toFixed(2)}
            </Badge>
            <Badge
              style={{
                backgroundColor: ACCENT,
                color: "var(--color-background, #fff)",
              }}
            >
              true quality {current.trueQuality.toFixed(2)}
            </Badge>
            <Badge variant="outline">KL {current.kl.toFixed(2)}</Badge>
            <Badge variant={hacking > 0.18 ? "destructive" : "outline"}>
              {hacking > 0.18 ? "⚠ reward hacking" : "✓ aligned"} gap{" "}
              {hacking.toFixed(2)}
            </Badge>
          </div>
          <div className="ring-foreground/10 rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground">
                <tr>
                  <th className="px-2 py-1.5 font-medium">style</th>
                  <th className="px-2 py-1.5 text-right font-medium">
                    policy %
                  </th>
                  <th className="px-2 py-1.5 text-right font-medium">
                    RM score
                  </th>
                  <th className="px-2 py-1.5 text-right font-medium">
                    true quality
                  </th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {styles.map((style, i) => (
                  <tr
                    key={style.name}
                    className={
                      i === inspect
                        ? "bg-foreground/5"
                        : i === argmax
                          ? "text-foreground"
                          : "text-muted-foreground"
                    }
                  >
                    <td className="px-2 py-1.5">{style.name}</td>
                    <td className="px-2 py-1.5 text-right">
                      {percent(current.pi[i] ?? 0)}
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      {(current.rmScores[i] ?? 0).toFixed(2)}
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      {style.trueQuality.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted-foreground text-xs">
            {styles[inspect]?.blurb}
          </p>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Adjust a slider, corrupt the reward, or load a preset to start the log."
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
        Each step runs one closed-form gradient-ascent update on the
        paper&apos;s RL objective: proxy reward minus beta times the KL from the
        SFT model plus gamma times the pretraining term. The response space is
        collapsed to {styles.length} named styles so the loop is legible; the
        real model optimizes over every possible token sequence with a 6B reward
        model. True quality is the hidden human preference the reward model only
        estimates; the gap between the two lines is the reward hacking the paper
        guards against. The run is deterministic — every preset produces the
        same trajectory on every load.
      </p>
    </div>
  );
}
