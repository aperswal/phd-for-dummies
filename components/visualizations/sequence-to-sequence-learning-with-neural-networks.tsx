"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  computeSeq,
  EOS,
  getSentence,
  initialState,
  PRESETS,
  SENTENCES,
  step,
  totalSteps,
  UNK,
  viewAtCursor,
  type Intervention,
  type SeqState,
  type SimEvent,
} from "@/components/visualizations/sequence-to-sequence-learning-with-neural-networks-model";

// Clay-orange marks the currently-active token being processed. Sage marks
// successfully recovered output words. These two meanings never overlap so the
// same color is never ambiguous within one view.
const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#8a9a6b";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SeqState, action: ViewAction): SeqState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

interface SliderProps {
  label: string;
  ariaLabel: string;
  value: number;
  min: number;
  max: number;
  stepBy: number;
  display: string;
  onChange: (value: number) => void;
}

function Slider({
  label,
  ariaLabel,
  value,
  min,
  max,
  stepBy,
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
        step={stepBy}
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

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1">
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
  );
}

// A scroll container that shows a gradient fade at the bottom when there is
// hidden overflow, so the reader knows more content is below the fold.
function FadeScroll({
  className,
  children,
  scrollRef,
}: {
  className?: string;
  children: React.ReactNode;
  scrollRef?: React.RefObject<HTMLDivElement | null>;
}) {
  const ownRef = useRef<HTMLDivElement>(null);
  const ref = scrollRef ?? ownRef;
  const [hasOverflow, setHasOverflow] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const check = () => setHasOverflow(el.scrollHeight > el.clientHeight + 2);
    check();
    const observer = new ResizeObserver(check);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);

  return (
    <div className="relative">
      <div
        ref={ref}
        className={`overflow-y-auto ${className ?? ""}`}
        style={{ scrollbarWidth: "thin" }}
      >
        {children}
      </div>
      {hasOverflow && (
        <div
          className="pointer-events-none absolute right-0 bottom-0 left-0 h-6 rounded-b-lg"
          style={{
            background:
              "linear-gradient(to bottom, transparent, var(--color-card, white))",
          }}
        />
      )}
    </div>
  );
}

function EventLog({
  events,
  onRestore,
}: {
  events: SimEvent<SeqState["params"]>[];
  onRestore: (event: SimEvent<SeqState["params"]>) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Newest events are shown first (reversed list), so when a new event arrives
  // scroll to top so the reader sees it immediately.
  useEffect(() => {
    if (scrollRef.current && events.length > 0) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length]);

  return (
    <FadeScroll
      className="ring-foreground/10 max-h-28 rounded-lg ring-1"
      scrollRef={scrollRef}
    >
      {events.length === 0 ? (
        <p className="text-muted-foreground px-2 py-1.5 text-xs">
          Flip a control or load a preset to start the log.
        </p>
      ) : (
        <ul className="text-xs">
          {[...events].reverse().map((event) => (
            <li key={event.id}>
              <button
                type="button"
                className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                onClick={() => onRestore(event)}
              >
                {event.id}. {event.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </FadeScroll>
  );
}

const FIXED_INTERVAL_MS = 900;
const MAX_STEPS_PER_FRAME = 60;

export function Seq2SeqRelay() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("forward-long"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("forward-long");

  const { params, log, cursor } = state;
  const result = useMemo(() => computeSeq(params), [params]);
  const sentence = getSentence(params.sentenceId);
  const total = totalSteps(result);
  const view = viewAtCursor(result, cursor);

  const accumulator = useRef(0);
  const intervalMs = FIXED_INTERVAL_MS / speed;

  const resetClock = useCallback(() => {
    accumulator.current = 0;
  }, []);

  // Fixed-timestep loop built on the shared animation-frame hook, so the run is
  // frame-rate independent. Stops playback automatically when the run is done
  // so the terminal state is clearly reached and the play button stops spinning.
  useAnimationFrame(playing, (deltaMs) => {
    if (view.phase === "done") {
      setPlaying(false);
      return;
    }
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < MAX_STEPS_PER_FRAME) {
      accumulator.current -= intervalMs;
      dispatch({ kind: "tick" });
      steps += 1;
    }
  });

  // Scroll the trace table so the row currently being decoded is always visible.
  const traceScrollRef = useRef<HTMLDivElement>(null);
  const activeRowRef = useRef<HTMLTableRowElement>(null);
  useEffect(() => {
    if (activeRowRef.current && traceScrollRef.current) {
      activeRowRef.current.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [view.activeDecodeIndex]);

  const intervene = useCallback(
    (action: Intervention) => {
      resetClock();
      dispatch({ kind: "intervene", action });
    },
    [resetClock],
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

  // Layout. A top row shows the source words in the order the encoder reads them,
  // a single jar in the middle, and the decoder's output stream below. Words
  // light up as the cursor reaches them.
  const width = 1000;
  const margin = 30;
  const jarX = width / 2;
  const jarY = 95;
  const srcY = 30;
  const outY = 175;
  const n = result.fedOrder.length;
  const srcSlot = (width - margin * 2) / Math.max(n, 1);
  const outCount = result.emitted.length;
  const outSlot = (width - margin * 2) / Math.max(outCount, 1);

  const presetBlurb = PRESETS.find((p) => p.id === presetId)?.blurb;

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
      <p className="text-muted-foreground text-sm">{presetBlurb}</p>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} 230`}
          className="w-full"
          role="img"
          aria-label={`A source sentence read ${
            params.reversed ? "reversed" : "forward"
          } into one memory jar, then decoded into output words. ${
            result.recoveredCount
          } of ${result.target.length} words recovered.`}
        >
          {/* Source words, in the fed order. The active one (clay-orange fill)
              is being written now; already-written ones fade to soft orange. */}
          {result.fedOrder.map((word, i) => {
            const x = margin + i * srcSlot + 3;
            const w = srcSlot - 6;
            const written = i < view.encodeProgress;
            const active = view.activeEncodeIndex === i;
            return (
              <g key={`src-${i}`}>
                <rect
                  x={x}
                  y={srcY - 16}
                  width={w}
                  height={32}
                  rx={6}
                  fill={
                    active
                      ? ACCENT
                      : written
                        ? ACCENT_SOFT
                        : "var(--color-muted)"
                  }
                  fillOpacity={written || active ? 1 : 0.55}
                />
                <text
                  x={x + w / 2}
                  y={srcY + 5}
                  textAnchor="middle"
                  fill={
                    active
                      ? "var(--color-primary-foreground, #fff)"
                      : "var(--color-foreground)"
                  }
                  style={{
                    fontSize: Math.min(14, srcSlot * 0.4),
                    fontWeight: 500,
                  }}
                >
                  {word}
                </text>
                {written && (
                  <line
                    x1={x + w / 2}
                    y1={srcY + 16}
                    x2={jarX}
                    y2={jarY - 18}
                    stroke={ACCENT}
                    strokeWidth={0.8}
                    strokeOpacity={0.35}
                  />
                )}
              </g>
            );
          })}
          <text
            x={margin}
            y={srcY - 24}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            source read {params.reversed ? "reversed ->" : "forward ->"}{" "}
            {result.fedOrder.length} words
          </text>

          {/* The single fixed-size memory jar. */}
          <g>
            <rect
              x={jarX - 34}
              y={jarY - 18}
              width={68}
              height={40}
              rx={9}
              fill={ACCENT}
              fillOpacity={0.18 + 0.5 * (view.encodeProgress / Math.max(n, 1))}
              stroke={ACCENT}
              strokeWidth={2}
            />
            <text
              x={jarX}
              y={jarY + 6}
              textAnchor="middle"
              fill={ACCENT}
              style={{ fontSize: 12, fontWeight: 600 }}
            >
              memory
            </text>
            {result.capacityPressure > 0 && (
              <text
                x={jarX}
                y={jarY + 32}
                textAnchor="middle"
                fill={ACCENT}
                style={{ fontSize: 10 }}
              >
                overfull {percent(result.capacityPressure)}
              </text>
            )}
          </g>

          {/* The decoder's output stream. Each emitted word lights up in turn.
              Sage (green) = recovered correctly. Clay-orange = UNK (failed).
              Each output token also carries a word label so color is never
              the only signal distinguishing success from failure. */}
          {result.emitted.map((word, i) => {
            const x = margin + i * outSlot + 3;
            const w = outSlot - 6;
            const done = i < view.decodeProgress;
            const active = view.activeDecodeIndex === i;
            const isUnk = word === UNK;
            const isEos = word === EOS;
            const fill = isEos ? "var(--color-muted)" : isUnk ? ACCENT : SAGE;
            return (
              <g key={`out-${i}`}>
                {(done || active) && (
                  <line
                    x1={jarX}
                    y1={jarY + 22}
                    x2={x + w / 2}
                    y2={outY - 16}
                    stroke={isUnk ? ACCENT : SAGE}
                    strokeWidth={1}
                    strokeOpacity={0.4}
                  />
                )}
                <rect
                  x={x}
                  y={outY - 16}
                  width={w}
                  height={32}
                  rx={6}
                  fill={done || active ? fill : "var(--color-muted)"}
                  fillOpacity={done || active ? (isEos ? 0.7 : 0.85) : 0.4}
                  stroke={
                    active ? "var(--color-foreground)" : "transparent"
                  }
                  strokeOpacity={0.4}
                  strokeWidth={1.5}
                />
                <text
                  x={x + w / 2}
                  y={outY + 5}
                  textAnchor="middle"
                  fill={
                    (done || active) && !isEos
                      ? "var(--color-primary-foreground, #fff)"
                      : "var(--color-foreground)"
                  }
                  fillOpacity={done || active ? 1 : 0.5}
                  style={{
                    fontSize: Math.min(13, outSlot * 0.4),
                    fontWeight: 500,
                  }}
                >
                  {word}
                </text>
              </g>
            );
          })}
          <text
            x={margin}
            y={outY + 30}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            decoder output — green = recovered, orange = UNK (lost)
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
        <span className="text-muted-foreground text-xs">
          step {cursor} / {total} &middot; {view.phase}
        </span>
        {/* Speed controls playback pace only — it does not change the model's
            signal-decay or reversal logic. */}
        <div className="min-w-32 max-w-48 flex-1">
          <Slider
            label="Playback speed"
            ariaLabel="Playback speed — controls how fast the animation steps through the run"
            value={speed}
            min={0.5}
            max={3}
            stepBy={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            {/* The paper's central trick: reversing the source shrinks the lag
                for early words, making training easier. */}
            <Toggle
              label={`Source order: ${params.reversed ? "reversed ✓" : "forward"}`}
              active={params.reversed}
              onClick={() => intervene({ type: "toggleReversed" })}
            />
            <Toggle
              label={`Stop token: ${params.dropEos ? "dropped" : "on ✓"}`}
              active={params.dropEos}
              onClick={() => intervene({ type: "toggleDropEos" })}
            />
          </div>
          <div className="flex flex-wrap gap-1.5">
            {SENTENCES.map((item) => (
              <Toggle
                key={item.id}
                label={item.label}
                active={params.sentenceId === item.id}
                onClick={() => intervene({ type: "setSentence", id: item.id })}
              />
            ))}
          </div>
          <p className="text-muted-foreground text-xs">{sentence.note}</p>
          {/* Memory tau: how many recurrent steps a signal survives before
              fading. This is the lag vs. signal tradeoff that reversal shortens. */}
          <Slider
            label="Memory tau — steps a signal survives"
            ariaLabel="Memory tau: controls how many recurrent steps a word's signal survives before fading"
            value={params.memory}
            min={1}
            max={30}
            stepBy={0.5}
            display={params.memory.toFixed(1)}
            onChange={(value) => intervene({ type: "setMemory", value })}
          />
          {/* Jar capacity: sentences longer than this attenuate every stored
              signal, illustrating the fixed-vector bottleneck. */}
          <Slider
            label="Jar capacity — words before the vector overfills"
            ariaLabel="Jar capacity: sentences longer than this attenuate all stored signals, showing the fixed-vector bottleneck"
            value={params.capacity}
            min={2}
            max={14}
            stepBy={1}
            display={String(params.capacity)}
            onChange={(value) => intervene({ type: "setCapacity", value })}
          />
        </Panel>

        <Panel title="Inside the run">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant={result.allRecovered ? "secondary" : "destructive"}>
              {result.allRecovered ? "all words recovered" : "words lost"}
            </Badge>
            <Badge variant="outline">
              recovered {result.recoveredCount}/{result.target.length}
            </Badge>
            <Badge variant="outline">accuracy {percent(result.accuracy)}</Badge>
            {result.runaway && <Badge variant="destructive">never stops</Badge>}
          </div>
          {/* Trace table scrolls to follow the active decode row during playback. */}
          <FadeScroll
            className="ring-foreground/10 max-h-44 rounded-lg ring-1"
            scrollRef={traceScrollRef}
          >
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">source</th>
                  <th className="px-2 py-1 font-medium">target</th>
                  <th className="px-2 py-1 text-right font-medium">lag</th>
                  <th className="px-2 py-1 text-right font-medium">signal</th>
                  <th className="px-2 py-1 font-medium">emitted</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {result.traces.map((trace) => {
                  const isActiveRow =
                    view.activeDecodeIndex === trace.targetIndex;
                  return (
                    <tr
                      key={`row-${trace.targetIndex}`}
                      ref={isActiveRow ? activeRowRef : null}
                      className={
                        trace.recovered
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }
                    >
                      <td className="px-2 py-1">{trace.sourceWord}</td>
                      <td className="px-2 py-1">{trace.targetWord}</td>
                      <td className="px-2 py-1 text-right">{trace.lag}</td>
                      <td className="px-2 py-1 text-right">
                        {trace.signal.toFixed(3)}
                      </td>
                      {/* emitted column: word label carries the meaning;
                          color reinforces it but is never the only signal. */}
                      <td
                        className="px-2 py-1"
                        style={{ color: trace.recovered ? SAGE : ACCENT }}
                      >
                        {trace.emitted}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </FadeScroll>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLog
              events={log.events}
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  params: event.snapshot,
                  label: `rewind to step ${event.id}`,
                })
              }
            />
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        A target word comes out right only when the signal from its matching
        source word survives the lag between when that word entered the jar and
        when its translation leaves, modeled here as exp(-lag / tau). Reversing
        the source flips the read order, so the first source word is written
        last and its target follows one step later. A trained LSTM learns this
        carrying from data; here the decay law is fixed so the pattern is
        legible. The recovery threshold is fixed at 0.12 — a signal must reach
        at least this level to count as recovered. This runs sentences of up to{" "}
        {SENTENCES.at(-1)?.source.length ?? 9} words; the paper&apos;s LSTM ran
        4 layers of 1000 cells over sentences of any length.
      </p>
    </div>
  );
}
