"use client";

import { useCallback, useMemo, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  computeAlignment,
  getPair,
  initialState,
  PAIRS,
  PRESETS,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/neural-machine-translation-by-jointly-learning-to-align-and-translate-model";

const ACCENT = "#c2683f";
// Non-argmax bars use ACCENT at 35% opacity so they remain visible in both
// light and dark mode without a hardcoded light-only hex.
const ACCENT_BAR_OPACITY = 0.35;

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
        aria-label={label}
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

export function AlignAndTranslate() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("monotonic"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("monotonic");
  const [inspect, setInspect] = useState<number | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const { params, log } = state;
  const result = useMemo(() => computeAlignment(params), [params]);
  const pair = getPair(params.pairId);
  const source = result.source;

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
    setInspect(null);
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

  // SVG layout. The source sentence runs along the top row; the word being
  // written sits centered below; arcs bow down from each source word to the
  // output, thickness set by the alignment weight; weight bars hang under the
  // source row.
  const width = 1000;
  const margin = 28;
  const slot = (width - margin * 2) / source.length;
  const sourceY = 60;
  const tokenH = 38;
  const barTop = 92;
  const barMax = 80;
  const outputY = 245;
  const centerX = (i: number) => margin + (i + 0.5) * slot;
  const outputX = width / 2;
  const argmax = result.argmax;

  // Show the scroll fade only for long sentences (> 6 words) whose rows
  // exceed the max-h-44 cap.
  const showScrollFade = source.length > 6;

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
          viewBox={`0 0 ${width} 300`}
          className="w-full"
          role="img"
          aria-label={`Writing the word '${result.target[result.step]}'. The model reads hardest from the source word '${source[argmax]}'${result.bottleneck ? ", but the fixed-vector bottleneck flattens the read across all source words" : ""}.`}
        >
          {source.map((_, j) => {
            const weight = result.weights[j] ?? 0;
            const x1 = centerX(j);
            return (
              <path
                key={`arc-${j}`}
                d={`M ${x1} ${sourceY + tokenH / 2} Q ${(x1 + outputX) / 2} ${(sourceY + outputY) / 2 + 30} ${outputX} ${outputY - tokenH / 2}`}
                fill="none"
                stroke={ACCENT}
                strokeWidth={1 + weight * 16}
                strokeOpacity={0.15 + weight * 0.72}
                strokeLinecap="round"
              />
            );
          })}

          {source.map((token, j) => {
            const x = margin + j * slot + 3;
            const w = slot - 6;
            const isTop = j === argmax;
            return (
              <g key={`src-${j}`}>
                <rect
                  x={x}
                  y={sourceY - tokenH / 2}
                  width={w}
                  height={tokenH}
                  rx={7}
                  fill="var(--color-muted)"
                  stroke={isTop ? ACCENT : "transparent"}
                  strokeWidth={2}
                  strokeDasharray={isTop ? "4 3" : undefined}
                />
                <text
                  x={x + w / 2}
                  y={sourceY + 4}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{
                    fontSize: Math.min(15, slot * 0.32),
                    fontWeight: 500,
                  }}
                >
                  {token}
                </text>
              </g>
            );
          })}

          {source.map((_, j) => {
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
                  fill={ACCENT}
                  fillOpacity={j === argmax ? 1 : ACCENT_BAR_OPACITY}
                />
                {weight > 0.04 && (
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

          <g>
            <rect
              x={outputX - slot / 2}
              y={outputY - tokenH / 2}
              width={slot}
              height={tokenH}
              rx={7}
              fill={ACCENT}
            />
            <text
              x={outputX}
              y={outputY + 4}
              textAnchor="middle"
              fill="var(--color-primary-foreground, #fff)"
              style={{ fontSize: Math.min(15, slot * 0.32), fontWeight: 600 }}
            >
              {result.target[result.step]}
            </text>
          </g>

          <text
            x={margin}
            y={barTop + barMax + 18}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            {result.bottleneck
              ? "fixed summary: same flat read every word"
              : "alignment weight per source word"}
          </text>
        </svg>
      </div>

      <div className="ring-foreground/10 flex flex-col gap-2 rounded-xl px-4 py-3 ring-1">
        <span className="text-muted-foreground text-xs font-medium">
          Translation so far
        </span>
        <p className="flex flex-wrap gap-1.5 text-sm">
          {result.target.map((word, i) => (
            <span
              key={`out-${i}`}
              className={
                i === result.step
                  ? "text-foreground rounded px-1.5 py-0.5 font-medium ring-1"
                  : i < result.step
                    ? "text-foreground px-1.5 py-0.5"
                    : "text-muted-foreground/40 px-1.5 py-0.5"
              }
              style={
                i === result.step
                  ? { color: ACCENT, borderColor: ACCENT }
                  : undefined
              }
            >
              {word}
            </span>
          ))}
        </p>
        <p className="text-muted-foreground text-xs">
          source:{" "}
          <span className="text-foreground">{pair.source.join(" ")}</span>
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="min-w-[8rem] max-w-[12rem] flex-1">
          <Slider
            label="Playback speed (how fast the animation steps)"
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
            {PAIRS.map((item) => (
              <Toggle
                key={item.id}
                label={item.id}
                active={params.pairId === item.id}
                onClick={() => intervene({ type: "setPair", id: item.id })}
              />
            ))}
          </div>
          <p className="text-muted-foreground text-xs">{pair.note}</p>
          <Toggle
            label="Fixed-vector bottleneck"
            active={params.bottleneck}
            onClick={() => intervene({ type: "toggleBottleneck" })}
          />
          <p className="text-muted-foreground text-xs">
            {params.bottleneck
              ? "On: one fixed summary for every word, the old encoder-decoder."
              : "Off: the decoder re-reads the source for each word, the paper's RNNsearch."}
          </p>
          <Slider
            label="Alignment sharpness — how focused each read is (v_a: higher = one dominant source word; lower = weight spread evenly)"
            value={params.sharpness}
            min={0.5}
            max={10}
            step={0.5}
            display={params.sharpness.toFixed(1)}
            onChange={(value) => intervene({ type: "setSharpness", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the alignment">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              writing &apos;{result.target[result.step]}&apos;
            </Badge>
            <Badge variant="outline">
              weights sum {result.weights.reduce((s, v) => s + v, 0).toFixed(2)}
            </Badge>
            <Badge variant="outline">
              reads &apos;{source[argmax]}&apos; {percent(result.maxWeight)}
            </Badge>
            <Badge variant="outline">
              spread {result.entropy.toFixed(2)} nats
            </Badge>
          </div>
          <div className="relative">
            <div
              ref={tableRef}
              className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1"
            >
              <table className="w-full text-left text-xs">
                <thead className="bg-card text-muted-foreground sticky top-0">
                  <tr>
                    <th className="px-2 py-1 font-medium">source word</th>
                    <th className="px-2 py-1 text-right font-medium">energy</th>
                    <th className="px-2 py-1 text-right font-medium">weight</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {source.map((token, j) => (
                    <tr
                      key={`row-${j}`}
                      className={
                        j === inspect
                          ? "bg-foreground/5"
                          : j === argmax
                            ? "text-foreground"
                            : "text-muted-foreground"
                      }
                      onMouseEnter={() => setInspect(j)}
                      onFocus={() => setInspect(j)}
                    >
                      <td className="px-2 py-1">{token}</td>
                      <td className="px-2 py-1 text-right">
                        {(result.energies[j] ?? 0).toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {percent(result.weights[j] ?? 0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Fade mask — visible only when the table can scroll (long sentence). */}
            {showScrollFade && (
              <div
                className="pointer-events-none absolute bottom-0 left-0 right-0 h-6 rounded-b-lg"
                style={{
                  background:
                    "linear-gradient(to bottom, transparent, var(--color-card))",
                }}
                aria-hidden="true"
              />
            )}
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step, switch a sentence, or flip the bottleneck to start the log."
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
        The alignment math here is the paper&apos;s additive model, energy{" "}
        <span className="font-mono">e = v_a&middot;tanh(W_a s + U_a h)</span>{" "}
        softmaxed over the source into weights that sum to one, then the context{" "}
        <span className="font-mono">c = &Sigma; &alpha; h</span>. The source
        annotations and what each output word reaches for are hand-set so the
        alignments are legible, where a trained model learns those from data.
        The run is fully deterministic. This runs a few short sentences; the
        paper&apos;s model runs 1000-unit RNNs over sentences of 30 to 50 words.
      </p>
    </div>
  );
}
