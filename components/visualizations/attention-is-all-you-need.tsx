"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  computeAttention,
  getSentence,
  HEADS,
  initialState,
  PRESETS,
  SENTENCES,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/attention-is-all-you-need-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
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

export function AttentionFlow() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("meaning"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("meaning");
  const [inspect, setInspect] = useState<number | null>(null);

  const { params, log } = state;
  const result = useMemo(() => computeAttention(params), [params]);
  const sentence = getSentence(params.sentenceId);
  const tokens = result.tokens;

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 1150 / speed, () =>
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

  // SVG layout. The sentence runs along one row; arcs bow up from the query word
  // to every key, thickness set by the attention weight; weight bars hang below.
  const width = 1000;
  const margin = 28;
  const slot = (width - margin * 2) / tokens.length;
  const tokenY = 150;
  const tokenH = 40;
  const barTop = 215;
  const barMax = 95;
  const centerX = (i: number) => margin + (i + 0.5) * slot;
  const masked = (j: number) => params.causalMask && j > result.queryIndex;
  const argmax = result.weights.indexOf(Math.max(...result.weights));

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
          aria-label={`Attention from the word '${tokens[result.queryIndex]}' to every word in the sentence. The strongest weight is on '${tokens[argmax]}'.`}
        >
          {tokens.map((_, j) => {
            if (j === result.queryIndex || masked(j)) return null;
            const weight = result.weights[j] ?? 0;
            const x1 = centerX(result.queryIndex);
            const x2 = centerX(j);
            const lift = 60 + Math.abs(x2 - x1) * 0.18;
            return (
              <path
                key={`arc-${j}`}
                d={`M ${x1} ${tokenY} Q ${(x1 + x2) / 2} ${tokenY - lift} ${x2} ${tokenY}`}
                fill="none"
                stroke={ACCENT}
                strokeWidth={1 + weight * 16}
                strokeOpacity={0.18 + weight * 0.72}
                strokeLinecap="round"
              />
            );
          })}

          {tokens.map((token, j) => {
            const isQuery = j === result.queryIndex;
            const isMasked = masked(j);
            const x = margin + j * slot + 3;
            const w = slot - 6;
            return (
              <g
                key={`tok-${j}`}
                role="button"
                tabIndex={0}
                aria-label={`Set query to ${token}`}
                aria-pressed={isQuery}
                className="cursor-pointer outline-none"
                onClick={() => intervene({ type: "setQuery", index: j })}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    intervene({ type: "setQuery", index: j });
                  }
                }}
                onFocus={() => setInspect(j)}
                onMouseEnter={() => setInspect(j)}
              >
                <rect
                  x={x}
                  y={tokenY - tokenH / 2}
                  width={w}
                  height={tokenH}
                  rx={7}
                  fill={isQuery ? ACCENT : "var(--color-muted)"}
                  fillOpacity={isMasked ? 0.25 : 1}
                  stroke={j === argmax && !isQuery ? ACCENT : "transparent"}
                  strokeWidth={2}
                  strokeDasharray={j === argmax && !isQuery ? "4 3" : undefined}
                />
                <text
                  x={x + w / 2}
                  y={tokenY + 5}
                  textAnchor="middle"
                  className="fill-foreground"
                  fill={isQuery ? "#fff" : undefined}
                  fillOpacity={isMasked ? 0.4 : 1}
                  style={{
                    fontSize: Math.min(15, slot * 0.34),
                    fontWeight: 500,
                  }}
                >
                  {token}
                </text>
                {isMasked && (
                  <text
                    x={x + w / 2}
                    y={tokenY - tokenH / 2 - 6}
                    textAnchor="middle"
                    fill={ACCENT}
                    style={{ fontSize: 11 }}
                  >
                    blocked
                  </text>
                )}
              </g>
            );
          })}

          {tokens.map((_, j) => {
            const weight = result.weights[j] ?? 0;
            const h = weight * barMax;
            const x = margin + j * slot + 3;
            const w = slot - 6;
            return (
              <g key={`bar-${j}`}>
                <rect
                  x={x}
                  y={barTop}
                  width={w}
                  height={barMax}
                  rx={3}
                  className="fill-foreground/5"
                />
                <rect
                  x={x}
                  y={barTop + (barMax - h)}
                  width={w}
                  height={h}
                  rx={3}
                  fill={j === argmax ? ACCENT : ACCENT_SOFT}
                />
                {weight > 0.025 && (
                  <text
                    x={x + w / 2}
                    y={barTop + (barMax - h) - 4}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 10 }}
                  >
                    {percent(weight)}
                  </text>
                )}
              </g>
            );
          })}
          <text
            x={margin}
            y={barTop + barMax + 18}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            attention weight per word
          </text>
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
        <SimPanel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            {HEADS.map((head, index) => (
              <Toggle
                key={head.name}
                label={head.name}
                active={params.headIndex === index}
                onClick={() => intervene({ type: "setHead", index })}
              />
            ))}
          </div>
          <p className="text-muted-foreground text-xs">
            {HEADS[params.headIndex]?.blurb}
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Scale by sqrt(dk)"
              active={params.scaling}
              onClick={() => intervene({ type: "toggleScaling" })}
            />
            <Toggle
              label="Causal mask"
              active={params.causalMask}
              onClick={() => intervene({ type: "toggleMask" })}
            />
          </div>
          <div className="flex flex-wrap gap-1.5">
            {SENTENCES.map((item) => (
              <Toggle
                key={item.id}
                label={item.id}
                active={params.sentenceId === item.id}
                onClick={() => intervene({ type: "setSentence", id: item.id })}
              />
            ))}
          </div>
          <Slider
            label="Score magnitude (grows with dk)"
            value={params.magnitude}
            min={1}
            max={20}
            step={0.5}
            display={params.magnitude.toFixed(1)}
            onChange={(value) => intervene({ type: "setMagnitude", value })}
          />
          <Slider
            label="Temperature"
            value={params.temperature}
            min={0.3}
            max={3}
            step={0.05}
            display={params.temperature.toFixed(2)}
            onChange={(value) => intervene({ type: "setTemperature", value })}
          />
          <Slider
            label="Seed (nudges the vectors)"
            value={params.seed}
            min={0}
            max={12}
            step={1}
            display={params.seed === 0 ? "off" : String(params.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the head">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              query &apos;{tokens[result.queryIndex]}&apos;
            </Badge>
            <Badge variant="outline">
              weights sum {result.weights.reduce((s, v) => s + v, 0).toFixed(2)}
            </Badge>
            <Badge variant="outline">
              top &apos;{tokens[argmax]}&apos; {percent(result.maxWeight)}
            </Badge>
            <Badge variant="outline">
              spread {result.entropy.toFixed(2)} nats
            </Badge>
          </div>
          <div className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">word</th>
                  <th className="px-2 py-1 text-right font-medium">score</th>
                  <th className="px-2 py-1 text-right font-medium">logit</th>
                  <th className="px-2 py-1 text-right font-medium">weight</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {tokens.map((token, j) => (
                  <tr
                    key={`row-${j}`}
                    className={
                      j === inspect
                        ? "bg-foreground/5"
                        : j === argmax
                          ? "text-foreground"
                          : "text-muted-foreground"
                    }
                  >
                    <td className="px-2 py-1">{token}</td>
                    <td className="px-2 py-1 text-right">
                      {(result.rawScores[j] ?? 0).toFixed(2)}
                    </td>
                    <td className="px-2 py-1 text-right">
                      {result.logits[j] !== undefined &&
                      Number.isFinite(result.logits[j])
                        ? (result.logits[j] ?? 0).toFixed(2)
                        : "-inf"}
                    </td>
                    <td className="px-2 py-1 text-right">
                      {percent(result.weights[j] ?? 0)}
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
              emptyHint="Step, click a word, or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  params: event.snapshot,
                  label: `rewind to step ${event.id}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The attention math here is exactly equation 1, softmax(Q&middot;K /
        sqrt(dk))&middot;V. The word vectors and the per-head projections are
        hand-set so the patterns are legible, where a trained model learns those
        numbers from data. This runs {HEADS.length} illustrative heads over{" "}
        {sentence.tokens.length} words; the paper&apos;s base model runs 8 heads
        of width 64 over thousands.
      </p>
    </div>
  );
}
