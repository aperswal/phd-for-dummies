"use client";

import type { ReactNode } from "react";
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
import { cn } from "@/lib/utils";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  initialState,
  isTerminal,
  masteredPoints,
  PRESETS,
  safetyHolds,
  step,
  type Intervention,
  type Reflection,
  type SimState,
} from "@/components/visualizations/reflexion-language-agents-with-verbal-reinforcement-learning-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
const SLATE = "#5b6470";

// Fixed playback pace: one model micro-step every 520 ms. Each tick is a
// discrete, meaningful phase (act / evaluate / reflect) that the reader should
// be able to observe individually, so there is no speed dial.
const TICK_INTERVAL_MS = 520;

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function reducer(state: SimState, action: ViewAction): SimState {
  return step(state, action);
}

const MAX_STEPS_PER_FRAME = 240;

// A fixed-timestep loop on top of the shared frame hook, kept local so this
// visualization stays self-contained. It advances the model once per whole
// interval regardless of device frame rate and caps catch-up after a stall.
function useFixedTimestep(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): () => void {
  const accumulator = useRef(0);
  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < MAX_STEPS_PER_FRAME) {
      accumulator.current -= intervalMs;
      onTick();
      steps += 1;
    }
  });
  return useCallback(() => {
    accumulator.current = 0;
  }, []);
}

function Panel({
  title,
  children,
  className,
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1",
        className,
      )}
    >
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
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

function Slider({
  label,
  value,
  min,
  max,
  step: stepSize,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
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

// Three discrete memory-window sizes (Omega = 1, 2, or 3) rendered as a
// segmented button group so the three distinct values are obvious rather than
// hidden inside a continuous range slider.
function MemoryWindowPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1 text-xs">
      <span className="text-muted-foreground">
        memory window (Omega) — how many lessons the agent holds at once
      </span>
      <div className="flex gap-1" role="group" aria-label="Memory window size">
        {([1, 2, 3] as const).map((size) => (
          <Button
            key={size}
            type="button"
            size="sm"
            variant={value === size ? "default" : "outline"}
            aria-pressed={value === size}
            onClick={() => onChange(size)}
            className="flex-1"
          >
            {size}
          </Button>
        ))}
      </div>
    </div>
  );
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

const PHASE_LABEL: Record<SimState["phase"], string> = {
  ready: "ready",
  acting: "actor choosing",
  evaluating: "evaluator scoring",
  reflecting: "writing reflection",
  passed: "passed",
  exhausted: "out of trials",
};

// One sparkline-style cell per trial showing pass / fail, so the reader watches
// the learning curve fill in. Truthful pass is sage, false-positive pass is
// clay (warning), fail is faint, the current trial pulses.
function TrialStrip({ state }: { state: SimState }) {
  const cells = Array.from({ length: state.params.maxTrials }, (_, trial) => {
    const outcome = state.outcomes.find((o) => o.trial === trial);
    const isCurrent =
      trial === state.trial && !isTerminal(state) && state.phase !== "ready";
    return { trial, outcome, isCurrent };
  });
  return (
    <div className="flex flex-wrap gap-1.5" aria-hidden>
      {cells.map(({ trial, outcome, isCurrent }) => {
        let fill = "transparent";
        let opacity = 0.18;
        let title = `trial ${trial}: not reached`;
        if (outcome) {
          if (outcome.passed && outcome.truthfulPass) {
            fill = SAGE;
            opacity = 1;
            title = `trial ${trial}: passed`;
          } else if (outcome.passed) {
            fill = ACCENT;
            opacity = 0.85;
            title = `trial ${trial}: false-positive pass`;
          } else {
            fill = SLATE;
            opacity = 0.5;
            title = `trial ${trial}: failed (${outcome.wrongPoints} wrong)`;
          }
        }
        return (
          <div
            key={trial}
            title={title}
            className={cn(
              "ring-foreground/15 size-5 rounded ring-1",
              isCurrent && "ring-foreground/60 ring-2",
            )}
            style={{ backgroundColor: fill, opacity }}
          />
        );
      })}
    </div>
  );
}

// Gradient fade at the bottom edge of a scrollable region: appears only when
// there is content below the fold so macOS overlay scrollbars (invisible at
// rest) do not silently hide overflow.
function ScrollFade({
  scrollRef,
}: {
  scrollRef: React.RefObject<HTMLDivElement | null>;
}) {
  const [canScrollDown, setCanScrollDown] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    function check() {
      if (!el) return;
      setCanScrollDown(el.scrollHeight - el.scrollTop > el.clientHeight + 2);
    }
    check();
    el.addEventListener("scroll", check);
    const observer = new ResizeObserver(check);
    observer.observe(el);
    return () => {
      el.removeEventListener("scroll", check);
      observer.disconnect();
    };
  }, [scrollRef]);

  if (!canScrollDown) return null;

  return (
    <div
      className="from-background pointer-events-none absolute right-0 bottom-0 left-0 h-6 rounded-b-lg bg-gradient-to-b to-transparent"
      aria-hidden
    />
  );
}

export function ReflexionLoop() {
  const [presetId, setPresetId] = useState("reflexion");
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("reflexion"),
  );
  const [playing, setPlaying] = useState(false);
  const [inspectId, setInspectId] = useState<number | null>(null);

  const { task, memory, params, trajectory, cursor, phase, outcomes } = state;

  const terminal = isTerminal(state);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // The animation gate: stop ticking once the run is done regardless of the
  // playing flag, so the fixedTimestep accumulator drains gracefully.
  const resetClock = useFixedTimestep(
    playing && !terminal,
    TICK_INTERVAL_MS,
    () => dispatch({ kind: "tick" }),
  );

  // Treat the run as paused once it reaches a terminal phase so the Play button
  // shows "Play" rather than "Pause" when the controls are re-enabled by Reset.
  const effectivePlaying = playing && !terminal;

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    setInspectId(null);
    dispatch({
      kind: "intervene",
      action: { type: "loadPreset", id: presetId },
    });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      setInspectId(null);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  const mastered = useMemo(() => masteredPoints(state), [state]);
  const safe = useMemo(() => safetyHolds(state), [state]);
  const passes = outcomes.filter((o) => o.passed).length;
  const inspected =
    inspectId !== null
      ? (memory.find((r) => r.id === inspectId) ?? null)
      : null;

  // Autoscroll the event log (newest-first list) to the top whenever a new
  // event is appended so the latest action is always visible without scrolling.
  const eventLogRef = useRef<HTMLDivElement>(null);
  const prevLogLength = useRef(state.log.length);
  useEffect(() => {
    if (state.log.length !== prevLogLength.current) {
      prevLogLength.current = state.log.length;
      if (eventLogRef.current) {
        eventLogRef.current.scrollTop = 0;
      }
    }
  }, [state.log.length]);

  // Autoscroll the memory list to the bottom (newest lesson appended at the
  // bottom) so the fresh lesson is visible as soon as it lands.
  const memoryListRef = useRef<HTMLDivElement>(null);
  const prevMemoryLength = useRef(memory.length);
  useEffect(() => {
    if (memory.length !== prevMemoryLength.current) {
      prevMemoryLength.current = memory.length;
      if (memoryListRef.current) {
        memoryListRef.current.scrollTop = memoryListRef.current.scrollHeight;
      }
    }
  }, [memory.length]);

  const cellPx = 116;
  const gapPx = 22;
  const totalW = task.length * cellPx + (task.length - 1) * gapPx;
  const svgH = 150;

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      {/* Preset buttons — the spine: four scenarios teaching the core contrast */}
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

      {/* The task as a row of decision points. The actor walks them left to
          right; each shows the choice made this trial and whether memory holds a
          lesson for it. */}
      <div className="ring-foreground/10 overflow-x-auto rounded-xl ring-1">
        <svg
          viewBox={`-16 -16 ${totalW + 32} ${svgH + 32}`}
          className="w-full min-w-[560px]"
          role="img"
          aria-label={`A row of ${task.length} decision points the agent must get right. The agent is at step ${cursor + 1}. It has mastered ${mastered} of ${task.length} points.`}
        >
          {task.map((point, index) => {
            const x = index * (cellPx + gapPx);
            const chosen = trajectory[index];
            const lesson = [...memory]
              .reverse()
              .find((r) => r.pointId === point.id);
            const decided = chosen !== undefined;
            const correctChoice = decided && chosen === point.correctAction;
            const isCursor =
              index === cursor && phase === "acting" && !terminal;
            const fill = lesson
              ? lesson.correct
                ? SAGE
                : ACCENT
              : "transparent";
            const fillOpacity = lesson ? (lesson.correct ? 0.18 : 0.16) : 0;
            return (
              <g key={point.id}>
                <rect
                  x={x}
                  y={8}
                  width={cellPx}
                  height={svgH - 16}
                  rx={12}
                  fill={fill}
                  fillOpacity={fillOpacity}
                  stroke={isCursor ? ACCENT : "var(--color-border)"}
                  strokeWidth={isCursor ? 3 : 1.25}
                />
                <text
                  x={x + cellPx / 2}
                  y={32}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                >
                  step {index + 1}
                </text>
                <text
                  x={x + cellPx / 2}
                  y={50}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{ fontSize: 12, fontWeight: 600 }}
                >
                  {point.label}
                </text>
                {decided && (
                  <text
                    x={x + cellPx / 2}
                    y={84}
                    textAnchor="middle"
                    fill={correctChoice ? SAGE : SLATE}
                    style={{ fontSize: 11, fontWeight: 600 }}
                  >
                    {correctChoice ? "correct" : "wrong"}
                  </text>
                )}
                {lesson ? (
                  <text
                    x={x + cellPx / 2}
                    y={svgH - 16}
                    textAnchor="middle"
                    fill={lesson.correct ? SAGE : ACCENT}
                    style={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
                  >
                    {lesson.correct ? "lesson: on target" : "lesson: misled"}
                  </text>
                ) : (
                  <text
                    x={x + cellPx / 2}
                    y={svgH - 16}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
                  >
                    no lesson yet
                  </text>
                )}
                {index < task.length - 1 && (
                  <line
                    x1={x + cellPx}
                    y1={svgH / 2}
                    x2={x + cellPx + gapPx}
                    y2={svgH / 2}
                    stroke="var(--color-border)"
                    strokeWidth={1.5}
                  />
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Transport — disabled entirely once the run is done so neither Play
          nor Step fires a no-op tick after a terminal phase. */}
      <div className="flex flex-wrap items-center gap-3">
        <PlayPauseStepControls
          playing={effectivePlaying}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={terminal}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="secondary">trial {state.trial}</Badge>
        <Badge variant="outline">{PHASE_LABEL[phase]}</Badge>
        <Badge variant="outline">
          mastered {mastered}/{task.length}
        </Badge>
        <Badge variant="outline">
          passes {passes}/{outcomes.length || 0}
        </Badge>
        {/* Color reinforced by word so meaning is never carried by color alone. */}
        <Badge
          variant={phase === "passed" ? "default" : "outline"}
          style={
            phase === "passed" && state.truthfulPass
              ? { backgroundColor: SAGE, color: "white" }
              : undefined
          }
        >
          {phase === "passed"
            ? state.truthfulPass
              ? "solved"
              : "passed (wrong answer)"
            : phase === "exhausted"
              ? "never solved"
              : "running"}
        </Badge>
        <Badge
          variant="outline"
          style={
            safe
              ? undefined
              : {
                  backgroundColor: ACCENT,
                  color: "white",
                  borderColor: ACCENT,
                }
          }
        >
          evaluator {safe ? "honest" : "passed a wrong answer"}
        </Badge>
      </div>

      <div className="flex flex-col gap-2">
        <span className="text-muted-foreground text-xs font-medium">
          Success across trials
        </span>
        <TrialStrip state={state} />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Perturb the loop">
          {/* Reflection toggle: the on/off switch for the paper's core mechanism */}
          <Toggle
            label={
              params.reflectionEnabled ? "Reflection: on" : "Reflection: off"
            }
            active={params.reflectionEnabled}
            onClick={() =>
              intervene({
                type: "setReflectionEnabled",
                value: !params.reflectionEnabled,
              })
            }
          />
          {/* Self-reflection accuracy: quality of the verbal lesson. 100% = the
              agent always names the right fix; 0% = always misdiagnoses. */}
          <Slider
            label="self-reflection accuracy — how often the lesson names the real mistake (100% = always, 0% = never)"
            value={params.reflectionAccuracy}
            min={0}
            max={1}
            step={0.02}
            display={pct(params.reflectionAccuracy)}
            onChange={(value) =>
              intervene({ type: "setReflectionAccuracy", value })
            }
          />
          {/* Evaluator false-positive rate: how often a wrong trajectory is
              declared correct. Recreates the MBPP flaky-test failure. */}
          <Slider
            label="evaluator false-positive rate — how often a wrong answer is declared correct (0% = trustworthy)"
            value={params.evaluatorFalsePositiveRate}
            min={0}
            max={1}
            step={0.05}
            display={pct(params.evaluatorFalsePositiveRate)}
            onChange={(value) =>
              intervene({ type: "setFalsePositiveRate", value })
            }
          />
          {/* Memory window (Omega): exactly 3 meaningful values matching the
              paper's bound of 1–3 lessons, shown as explicit buttons. */}
          <MemoryWindowPicker
            value={params.memoryCapacity}
            onChange={(value) =>
              intervene({ type: "setMemoryCapacity", value })
            }
          />
          <div className="flex flex-wrap gap-1.5">
            <Button
              type="button"
              size="sm"
              variant="secondary"
              disabled={!state.lastReflection || !state.lastReflection.correct}
              onClick={() => intervene({ type: "corruptLastReflection" })}
            >
              Corrupt last lesson
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              onClick={() => intervene({ type: "addTrap" })}
            >
              Add a decision point
            </Button>
          </div>
        </Panel>

        {/* Memory panel: the verbal reflections are the spine of the sim.
            Bounded height with a fade affordance; autoscrolls to the newest
            lesson (appended at the bottom). */}
        <Panel title="Memory (long-term)">
          {memory.length === 0 ? (
            <p className="text-muted-foreground text-xs">
              Empty. The agent fails the first trial blind, then the
              self-reflection model writes its first lesson here. Step until a
              fail, then watch a line appear.
            </p>
          ) : (
            <div className="relative">
              <div
                ref={memoryListRef}
                className="flex max-h-48 flex-col gap-1.5 overflow-y-auto"
              >
                {memory.map((reflection) => (
                  <button
                    key={reflection.id}
                    type="button"
                    onClick={() => setInspectId(reflection.id)}
                    className={cn(
                      "ring-foreground/10 hover:bg-foreground/5 w-full rounded-lg px-2 py-1.5 text-left text-xs ring-1",
                      inspectId === reflection.id &&
                        "ring-foreground/40 ring-2",
                    )}
                  >
                    <span className="flex items-center justify-between">
                      <span className="text-muted-foreground font-mono">
                        #{reflection.id} step {reflection.pointId + 1}
                      </span>
                      <span
                        style={{ color: reflection.correct ? SAGE : ACCENT }}
                      >
                        {reflection.correct ? "on target" : "misdiagnosis"}
                      </span>
                    </span>
                    <span className="text-foreground/80 mt-1 block">
                      {reflection.text}
                    </span>
                  </button>
                ))}
              </div>
              <ScrollFade scrollRef={memoryListRef} />
            </div>
          )}
          {inspected && (
            <div className="bg-foreground/5 rounded-lg p-2 text-xs">
              <p className="text-muted-foreground">
                The actor reads this before step {inspected.pointId + 1}. It
                will{" "}
                {inspected.correct
                  ? "take the suggested action and clear the step."
                  : "avoid the old action but the suggestion is still wrong, so it misses again."}
              </p>
            </div>
          )}
        </Panel>
      </div>

      {/* Event log: newest event at top (reversed list). Autoscrolls to the
          top on each new entry; fade affordance when overflow exists. */}
      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium">
          Event log
        </span>
        <div className="relative">
          <div
            ref={eventLogRef}
            className="ring-foreground/10 max-h-32 overflow-y-auto rounded-lg ring-1"
          >
            {state.log.length === 0 ? (
              <p className="text-muted-foreground px-2 py-1.5 text-xs">
                Press play or step to start the loop.
              </p>
            ) : (
              <ul className="text-xs">
                {[...state.log].reverse().map((event) => (
                  <li key={event.id}>
                    <button
                      type="button"
                      className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                      onClick={() =>
                        intervene({
                          type: "restore",
                          snapshot: event.snapshot,
                          label: `rewind to event ${event.id}`,
                        })
                      }
                    >
                      {event.id}. {event.label}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <ScrollFade scrollRef={eventLogRef} />
        </div>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs the paper&apos;s Algorithm 1 over a four-step task, not a real
        LLM. The actor, evaluator, and self-reflection model are stand-ins whose
        behavior follows the paper&apos;s rules. &quot;Mastered&quot; counts the
        steps the agent now gets right because a correct lesson sits in memory.
        A weight never changes; the only thing that updates between trials is
        the text in the memory panel. The run is deterministic: same preset,
        same trajectory every time.
      </p>
    </div>
  );
}

export type { Reflection };
