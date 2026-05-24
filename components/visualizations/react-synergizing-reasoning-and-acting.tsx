"use client";

import { useCallback, useReducer, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  getTask,
  grounding,
  initialState,
  PRESETS,
  step,
  TASKS,
  type Intervention,
  type Mode,
  type SimState,
  type Status,
  type StepKind,
  type TrajectoryStep,
} from "@/components/visualizations/react-synergizing-reasoning-and-acting-model";

const ACCENT = "#c2683f";
const SAGE = "#7d8a6a";

const MODE_LABELS: Record<Mode, string> = {
  react: "ReAct (think + act)",
  act: "Act only",
  cot: "CoT only",
};

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

// A fixed-timestep clock so the run advances at the same pace on any device. It
// accumulates elapsed ms and fires onTick once per whole interval, capping
// catch-up after a backgrounded tab.
function useFixedTimestep(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): () => void {
  const accumulator = useRef(0);
  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < 120) {
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
  children: React.ReactNode;
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

const STATUS_TEXT: Record<Status, string> = {
  running: "running",
  solved: "solved",
  wrong: "wrong answer",
  stuck: "stuck",
};

const STATUS_TONE: Record<Status, string> = {
  running: "bg-muted text-foreground",
  solved:
    "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  wrong: "bg-rose-100 text-rose-900 dark:bg-rose-950 dark:text-rose-200",
  stuck: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
};

function StepRow({ step: s }: { step: TrajectoryStep }) {
  if (s.index === 0) {
    return (
      <div className="bg-foreground/5 text-foreground rounded-lg px-3 py-2 text-sm font-medium">
        {s.text}
      </div>
    );
  }

  const tone: Record<StepKind, { label: string; cls: string; icon: string }> = {
    thought: {
      label: s.hallucinated ? "Thought (unverified)" : "Thought",
      cls: s.hallucinated
        ? "border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200"
        : "border-foreground/10 bg-card text-foreground",
      icon: "T",
    },
    action: {
      label: "Action",
      cls: "border-foreground/10 bg-card text-foreground",
      icon: "A",
    },
    observation: {
      label:
        s.informative === false
          ? "Observation (no useful info)"
          : "Observation",
      cls:
        s.informative === false
          ? "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
          : "border-foreground/10 bg-muted/40 text-foreground",
      icon: "O",
    },
  };

  const t = tone[s.kind];
  const grounded = s.kind === "thought" && s.grounded;

  return (
    <div
      className={cn("flex gap-2 rounded-lg border px-3 py-2 text-sm", t.cls)}
    >
      <span
        className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold"
        style={{
          backgroundColor:
            s.kind === "action"
              ? ACCENT
              : s.kind === "observation"
                ? SAGE
                : "transparent",
          color: s.kind === "thought" ? "currentColor" : "#fff",
          border: s.kind === "thought" ? "1px solid currentColor" : undefined,
        }}
        aria-hidden
      >
        {t.icon}
      </span>
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="flex items-center gap-1.5 text-[10px] font-medium tracking-wide uppercase opacity-70">
          {t.label}
          {grounded && (
            <span className="rounded-sm bg-emerald-200 px-1 text-[9px] text-emerald-900 dark:bg-emerald-900 dark:text-emerald-200">
              grounded
            </span>
          )}
        </span>
        <span className={cn(s.kind === "action" && "font-mono")}>{s.text}</span>
      </div>
    </div>
  );
}

export function ReActLoop() {
  const [presetId, setPresetId] = useState("react-grounded");
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("react-grounded"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [editText, setEditText] = useState("");

  const task = getTask(state.taskId);
  const g = grounding(state);
  const terminal = state.status !== "running";

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !terminal, 950 / speed, () =>
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

  const setMode = useCallback(
    (mode: Mode) => {
      setPlaying(false);
      resetClock();
      intervene({ type: "setMode", mode });
    },
    [intervene, resetClock],
  );

  const setTask = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      intervene({ type: "setTask", id });
    },
    [intervene, resetClock],
  );

  const preset = PRESETS.find((p) => p.id === presetId);
  const groundedPct = Math.round(g.ratio * 100);

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => (
          <Button
            key={p.id}
            type="button"
            size="sm"
            variant={presetId === p.id ? "default" : "outline"}
            aria-pressed={presetId === p.id}
            onClick={() => loadPreset(p.id)}
          >
            {p.label}
          </Button>
        ))}
      </div>
      <p className="text-muted-foreground text-sm">{preset?.blurb}</p>

      <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <Panel title="Trajectory" className="min-h-[20rem]">
          <div className="flex max-h-[24rem] flex-col gap-1.5 overflow-y-auto pr-1">
            {state.steps.map((s) => (
              <StepRow key={s.index} step={s} />
            ))}
            {terminal && state.finalAnswer && (
              <div
                className={cn(
                  "mt-1 rounded-lg px-3 py-2 text-sm font-semibold",
                  STATUS_TONE[state.status],
                )}
              >
                {state.status === "solved" ? "Answer: " : "Wrong answer: "}
                {state.finalAnswer}
              </div>
            )}
            {state.status === "stuck" && (
              <div
                className={cn(
                  "mt-1 rounded-lg px-3 py-2 text-sm font-semibold",
                  STATUS_TONE.stuck,
                )}
              >
                Stuck. No grounded answer reached.
              </div>
            )}
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Inside the run">
            <div className="flex flex-wrap gap-2 text-xs">
              <span
                className={cn(
                  "rounded-full px-2 py-0.5 font-medium",
                  STATUS_TONE[state.status],
                )}
              >
                {STATUS_TEXT[state.status]}
              </span>
              <Badge variant="secondary">{MODE_LABELS[state.mode]}</Badge>
              <Badge variant="outline">step {state.tick}</Badge>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-muted-foreground flex items-center justify-between text-xs">
                <span>grounded in observations</span>
                <span className="text-foreground font-mono">
                  {groundedPct}%
                </span>
              </span>
              <div className="bg-foreground/10 h-2 w-full overflow-hidden rounded-full">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${groundedPct}%`,
                    backgroundColor: groundedPct >= 50 ? SAGE : ACCENT,
                  }}
                />
              </div>
              <span className="text-muted-foreground text-[11px]">
                {g.grounded} fact{g.grounded === 1 ? "" : "s"} backed by a real
                observation, {g.memory} carried only in memory.
              </span>
            </div>
          </Panel>

          <Panel title="Mode">
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(MODE_LABELS) as Mode[]).map((mode) => (
                <Button
                  key={mode}
                  type="button"
                  size="sm"
                  variant={state.mode === mode ? "default" : "outline"}
                  aria-pressed={state.mode === mode}
                  onClick={() => setMode(mode)}
                >
                  {MODE_LABELS[mode]}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {TASKS.map((t) => (
                <Button
                  key={t.id}
                  type="button"
                  size="sm"
                  variant={state.taskId === t.id ? "secondary" : "ghost"}
                  aria-pressed={state.taskId === t.id}
                  onClick={() => setTask(t.id)}
                >
                  {t.label}
                </Button>
              ))}
            </div>
            <p className="text-muted-foreground text-[11px]">{task.note}</p>
          </Panel>

          <Panel title="Intervene mid-run">
            <div className="flex flex-wrap gap-1.5">
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={terminal || state.mode === "cot"}
                onClick={() => intervene({ type: "injectDeadSearch" })}
              >
                Kill next search
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={terminal}
                onClick={() => intervene({ type: "corruptMemory" })}
              >
                Corrupt memory
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={terminal}
                onClick={() => intervene({ type: "forceLoop" })}
              >
                Force a loop
              </Button>
            </div>
            <div className="flex flex-col gap-1.5">
              <label
                className="text-muted-foreground text-[11px]"
                htmlFor="thought-edit"
              >
                Edit the next thought, then step
              </label>
              <div className="flex gap-1.5">
                <input
                  id="thought-edit"
                  type="text"
                  value={editText}
                  placeholder="rewrite the agent's next thought"
                  onChange={(event) => setEditText(event.target.value)}
                  className="border-foreground/15 h-8 w-full rounded-md border bg-transparent px-2 text-xs outline-none focus-visible:ring-2"
                  style={{ ["--tw-ring-color" as string]: ACCENT }}
                />
                <Button
                  type="button"
                  size="sm"
                  disabled={terminal || editText.trim() === ""}
                  onClick={() => {
                    intervene({ type: "editThought", text: editText.trim() });
                    setEditText("");
                  }}
                >
                  Edit
                </Button>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={false}
        />
        <div className="flex items-center gap-3">
          <label className="flex w-40 flex-col gap-1 text-xs">
            <span className="text-muted-foreground flex items-center justify-between">
              <span>Speed</span>
              <span className="text-foreground font-mono">
                {speed.toFixed(1)}x
              </span>
            </span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
              className="h-1.5 w-full cursor-pointer"
              style={{ accentColor: ACCENT }}
            />
          </label>
          <label className="flex w-32 flex-col gap-1 text-xs">
            <span className="text-muted-foreground flex items-center justify-between">
              <span>Seed</span>
              <span className="text-foreground font-mono">{state.seed}</span>
            </span>
            <input
              type="range"
              min={0}
              max={20}
              step={1}
              value={state.seed}
              onChange={(event) =>
                intervene({
                  type: "setSeed",
                  value: Number(event.target.value),
                })
              }
              className="h-1.5 w-full cursor-pointer"
              style={{ accentColor: ACCENT }}
            />
          </label>
        </div>
      </div>

      <Panel title="Event log">
        <div className="ring-foreground/10 max-h-32 overflow-y-auto rounded-lg ring-1">
          {state.log.events.length === 0 ? (
            <p className="text-muted-foreground px-2 py-1.5 text-xs">
              Step the run, switch a mode, or inject a fault to start the log.
            </p>
          ) : (
            <ul className="text-xs">
              {[...state.log.events].reverse().map((event) => (
                <li
                  key={event.id}
                  className="text-muted-foreground px-2 py-1 font-mono"
                >
                  {event.id}. {event.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      </Panel>

      <p className="text-muted-foreground text-xs">
        The agent walks the paper&apos;s own HotpotQA and Fever trajectories one
        step at a time. The Wikipedia stub here returns a few canned pages,
        where the paper hits the live API; the reasoning, the action format
        (search/lookup/finish), and the failure modes follow the paper. Killing
        a search reproduces its search-error mode, corrupting memory reproduces
        the CoT hallucination mode, and editing a thought is the
        human-in-the-loop correction from Figure 5.
      </p>
    </div>
  );
}
