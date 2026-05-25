"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  GPT2_SIZE_INDEX,
  getTask,
  goldMatch,
  goldProbability,
  initialState,
  MODEL_SIZES,
  PRESETS,
  scoreCandidates,
  step,
  TASKS,
  type Candidate,
  type Intervention,
  type SimEvent,
  type SimState,
} from "@/components/visualizations/language-models-are-unsupervised-multitask-learners-model";

const ACCENT = "#c2683f";
const SAGE = "#7a8b6f";

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

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1">
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
  );
}

function EventLog({
  events,
  onRestore,
}: {
  events: SimEvent[];
  onRestore: (event: SimEvent) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Newest events appear at the top; scroll there whenever a new event is added.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length]);

  return (
    // Relative container so the fade mask can sit flush at the bottom edge.
    <div className="relative">
      <div
        ref={scrollRef}
        className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1"
      >
        {events.length === 0 ? (
          <p className="text-muted-foreground px-2 py-1.5 text-xs">
            Step, change the prompt, or toggle the hint to start the log.
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
      </div>
      {/* Fade mask signals more entries are hidden below when the list overflows. */}
      {events.length > 3 && (
        <div className="from-card pointer-events-none absolute right-0 bottom-0 left-0 h-5 rounded-b-lg bg-gradient-to-t to-transparent" />
      )}
    </div>
  );
}

function useFixedClock(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): () => void {
  const accumulator = useRef(0);
  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < 240) {
      accumulator.current -= intervalMs;
      onTick();
      steps += 1;
    }
  });
  return useCallback(() => {
    accumulator.current = 0;
  }, []);
}

export function ZeroShotGenerator() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("qa-happy"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("qa-happy");

  const { params, generated, finished, log } = state;
  const task = getTask(params.taskId);
  const size = MODEL_SIZES[params.sizeIndex] ?? MODEL_SIZES[GPT2_SIZE_INDEX];

  const candidates = useMemo(
    () => (finished ? [] : scoreCandidates(params, state.goldCursor)),
    [params, state.goldCursor, finished],
  );
  const live = candidates.filter((c) => !c.masked);
  const topCandidate = live.reduce<Candidate | null>(
    (best, c) => (best === null || c.prob > best.prob ? c : best),
    null,
  );
  const match = goldMatch(state);
  const gp = useMemo(
    () => goldProbability(params, state.goldCursor),
    [params, state.goldCursor],
  );

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedClock(playing && !finished, 900 / speed, () =>
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

  const promptText = params.hintPresent
    ? task.prompt
    : task.prompt.replace(task.hint, "").replace(/\s+$/, "");

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

      <div className="ring-foreground/10 flex flex-col gap-4 rounded-xl p-4 ring-1 sm:p-5">
        <div className="flex flex-col gap-1.5">
          <span className="text-muted-foreground text-xs font-medium">
            Prompt the model reads
            {!params.hintPresent && (
              <span className="ml-2" style={{ color: ACCENT }}>
                hint removed
              </span>
            )}
          </span>
          <pre className="bg-foreground/5 overflow-x-auto rounded-lg px-3 py-2 font-mono text-sm whitespace-pre-wrap">
            {promptText}
            <span className="text-foreground/40"> [generate]</span>
          </pre>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-muted-foreground text-xs font-medium">
            Generated, one token at a time
          </span>
          <div className="flex min-h-10 flex-wrap items-center gap-1.5">
            {generated.length === 0 && (
              <span className="text-muted-foreground text-sm">
                Press play or step to sample the next token.
              </span>
            )}
            {generated.map((token, i) => {
              const isGold = task.gold[i] === token;
              return (
                <span
                  key={`gen-${i}`}
                  className="rounded-md px-2 py-1 font-mono text-sm"
                  style={{
                    background: isGold ? ACCENT : "var(--color-muted)",
                    // Use a theme token so the label stays legible in dark mode.
                    color: isGold
                      ? "var(--color-primary-foreground)"
                      : undefined,
                  }}
                >
                  {token}
                </span>
              );
            })}
            {!finished && generated.length > 0 && (
              <span className="text-foreground/40 font-mono text-sm">|</span>
            )}
          </div>
          <p className="text-muted-foreground text-xs">
            Target answer{" "}
            <span className="text-foreground font-mono">
              {task.gold.join(" ")}
            </span>
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-muted-foreground text-xs font-medium">
            Next-token distribution{" "}
            {finished ? "(run finished)" : `(top ${params.topK} kept)`}
          </span>
          <div className="flex flex-col gap-1.5">
            {candidates.map((c) => {
              const width = c.masked ? 0 : c.prob * 100;
              return (
                <div
                  key={c.token}
                  className={cn(
                    "flex items-center gap-2 text-xs",
                    c.masked && "opacity-40",
                  )}
                >
                  <span className="w-28 truncate font-mono">
                    {c.token}
                    {c.isGold && (
                      // Star glyph so the gold token is identified by shape
                      // AND color, never color alone.
                      <span className="ml-1" style={{ color: ACCENT }}>
                        ★
                      </span>
                    )}
                  </span>
                  <div className="bg-foreground/5 relative h-4 flex-1 overflow-hidden rounded">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${width}%`,
                        background: c.isGold ? ACCENT : SAGE,
                      }}
                    />
                  </div>
                  <span className="text-muted-foreground w-12 text-right font-mono">
                    {c.masked ? "cut" : percent(c.prob)}
                  </span>
                </div>
              );
            })}
            {candidates.length === 0 && (
              <span className="text-muted-foreground text-xs">
                Reset or load a preset to sample again.
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          // Disable transport once the run is done so the terminal state is clear.
          disabled={finished}
        />
        <div className="w-40">
          <Slider
            label="Playback speed"
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
        <Panel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            {TASKS.map((item) => (
              <Toggle
                key={item.id}
                label={item.label}
                active={params.taskId === item.id}
                onClick={() => intervene({ type: "setTask", id: item.id })}
              />
            ))}
          </div>
          <p className="text-muted-foreground text-xs">{task.note}</p>
          <Toggle
            label={params.hintPresent ? "Task hint: on" : "Task hint: removed"}
            active={params.hintPresent}
            onClick={() => intervene({ type: "toggleHint" })}
          />
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Model size (capacity)
            </span>
            <div className="flex flex-wrap gap-1.5">
              {MODEL_SIZES.map((item, index) => (
                <Toggle
                  key={item.params}
                  label={item.params}
                  active={params.sizeIndex === index}
                  onClick={() => intervene({ type: "setSize", index })}
                />
              ))}
            </div>
          </div>
          <Slider
            label="Temperature — sharpens or flattens the sampling distribution"
            value={params.temperature}
            min={0.2}
            max={2}
            step={0.05}
            display={params.temperature.toFixed(2)}
            onChange={(value) => intervene({ type: "setTemperature", value })}
          />
          <Slider
            label="Top-k — how many candidate tokens the model keeps"
            value={params.topK}
            min={1}
            max={6}
            step={1}
            display={String(params.topK)}
            onChange={(value) => intervene({ type: "setTopK", value })}
          />
          {/* Seed is fixed per preset for reproducibility. Reshuffle advances
              it by a large prime so the reader can try a different sample
              without a meaningless 0–20 dial. */}
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => intervene({ type: "reshuffle" })}
          >
            Reshuffle sample
          </Button>
        </Panel>

        <Panel title="Inside the run">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{size?.params} model</Badge>
            <Badge variant="outline">p(correct next) {percent(gp)}</Badge>
            <Badge variant="outline">answer match {percent(match)}</Badge>
            <Badge variant={params.hintPresent ? "outline" : "destructive"}>
              hint {params.hintPresent ? "present" : "removed"}
            </Badge>
          </div>
          <div className="relative">
            <div className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1">
              <table className="w-full text-left text-xs">
                <thead className="bg-card text-muted-foreground sticky top-0">
                  <tr>
                    <th className="px-2 py-1 font-medium">token</th>
                    <th className="px-2 py-1 text-right font-medium">base</th>
                    <th className="px-2 py-1 text-right font-medium">boost</th>
                    <th className="px-2 py-1 text-right font-medium">prob</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {candidates.map((c) => (
                    <tr
                      key={`row-${c.token}`}
                      className={
                        c === topCandidate
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }
                    >
                      <td className="px-2 py-1">
                        {c.token}
                        {c.isGold && (
                          <span className="ml-1" style={{ color: ACCENT }}>
                            ★
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {c.base.toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {c.taskBoost.toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {c.masked ? "cut" : percent(c.prob)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Fade mask signals more rows are hidden below when the table overflows. */}
            {candidates.length > 5 && (
              <div className="from-card pointer-events-none absolute right-0 bottom-0 left-0 h-5 rounded-b-lg bg-gradient-to-t to-transparent" />
            )}
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLog
              events={log}
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
        The loop is real, the same predict-then-sample step the paper runs.
        Scores come from small hand-built per-task tables instead of a trained
        network, so the gold answer winning is legible rather than learned.
        Model size sets how hard the model leans toward the correct
        continuation, which is the log-linear lift of Figure 1, and the hint
        switch is the paper&apos;s {"TL;DR:"} ablation. Each preset starts from a
        fixed seed so every run is reproducible; use Reshuffle to try a
        different sample. This runs {TASKS.length} tasks over a handful of
        tokens; GPT-2 runs over a 50,257-token vocabulary and 40 GB of text.
      </p>
    </div>
  );
}
