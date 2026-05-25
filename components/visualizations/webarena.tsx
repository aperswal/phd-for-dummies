"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  evaluateRun,
  getResolvedPlan,
  getTask,
  initialState,
  PRESETS,
  reduce,
  TASKS,
  type Input,
  type Intervention,
  type Metric,
  type PlanName,
  type SimState,
} from "@/components/visualizations/webarena-model";

// ACCENT = clay-orange: fail / wrong / off-target (one meaning throughout)
const ACCENT = "#c2683f";
// SAGE = sage-green: pass / on-target / progress (one meaning throughout)
const SAGE = "#7d8c6a";
// SEED_COUNT must match the slider range in the model (0–11 inclusive)
const SEED_COUNT = 12;

const PLAN_LABELS: Record<PlanName, string> = {
  reference: "Reference path",
  alternative: "Alternative valid path",
  offTarget: "Off-target",
  giveUp: "Give up early",
};

function reducer(state: SimState, input: Input): SimState {
  return reduce(state, input);
}

// A titled panel, inlined so this visualization carries its own frame style.
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

function Toggle({
  label,
  active,
  onClick,
  disabled = false,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant={active ? "default" : "outline"}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
    >
      {label}
    </Button>
  );
}

function Light({
  on,
  pass,
  label,
}: {
  on: boolean;
  pass: boolean;
  label: string;
}) {
  const color = !on ? "var(--color-muted)" : pass ? SAGE : ACCENT;
  const symbol = !on ? "•" : pass ? "✓" : "✗";
  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        aria-hidden
        className="flex size-5 items-center justify-center rounded-full text-xs font-bold text-white"
        style={{ backgroundColor: color }}
      >
        {symbol}
      </span>
      <span className={on ? "text-foreground" : "text-muted-foreground"}>
        {label}
      </span>
    </div>
  );
}

export function WebArenaAgent() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState({ ...PRESETS[0]!.params }),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState(PRESETS[0]!.id);
  const accumulator = useRef(0);

  const { params } = state;
  const task = getTask(params.taskId);
  const resolved = useMemo(() => getResolvedPlan(params), [params]);

  const reward = useMemo(
    () => evaluateRun(task, state.actions, state.trajectory),
    [task, state.actions, state.trajectory],
  );

  const world = state.trajectory[state.trajectory.length - 1] ?? null;
  const totalActions = resolved.actions.length;
  const showResults = state.finished;

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [dispatch],
  );

  const resetClock = useCallback(() => {
    accumulator.current = 0;
  }, []);

  useAnimationFrame(playing, (deltaMs) => {
    // Stop auto-play once the run finishes so the loop doesn't spin on a done run.
    if (state.finished) {
      setPlaying(false);
      return;
    }
    accumulator.current += deltaMs;
    const interval = 900;
    while (accumulator.current >= interval) {
      accumulator.current -= interval;
      dispatch({ kind: "tick" });
    }
  });

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    intervene({ type: "restore", params, label: "reset run" });
  }, [intervene, params, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      intervene({ type: "loadPreset", id });
    },
    [intervene, resetClock],
  );

  const metric: Metric = params.metric;
  const visibleScore =
    metric === "functional" ? reward.functional : reward.surface;

  // Event log: scroll to top when a new entry appears (list is newest-first).
  const logScrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (logScrollRef.current) {
      logScrollRef.current.scrollTop = 0;
    }
  }, [state.log.length]);

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

      <div className="ring-foreground/10 rounded-xl ring-1">
        <div className="border-foreground/10 flex flex-col gap-1 border-b px-4 py-3">
          <span className="text-muted-foreground text-xs font-medium">
            {task.category} task
          </span>
          <span className="text-foreground text-sm">
            &ldquo;{task.intent}&rdquo;
          </span>
        </div>

        <div className="grid gap-0 sm:grid-cols-[1fr_1fr]">
          {/* The browser: current page plus the world the actions wrote to. */}
          <div className="border-foreground/10 flex flex-col gap-2 border-b px-4 py-3 sm:border-r sm:border-b-0">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">URL</span>
              <code className="bg-foreground/5 rounded px-2 py-0.5 text-xs">
                {world?.url ?? "home"}
              </code>
            </div>
            <div className="text-muted-foreground flex flex-col gap-1 text-xs">
              {world && world.cart.length > 0 && (
                <span>
                  cart:{" "}
                  <span className="text-foreground">
                    {world.cart.join(", ")}
                  </span>
                </span>
              )}
              {world && world.repoFiles["README"] !== undefined && (
                <span>
                  README:{" "}
                  <span className="text-foreground">
                    {world.repoFiles["README"]}
                  </span>
                </span>
              )}
              {world && world.posts.length > 0 && (
                <span>
                  posted:{" "}
                  <span className="text-foreground">
                    {world.posts.map((post) => post.body).join("; ")}
                  </span>
                </span>
              )}
              {world && world.answer !== null && (
                <span>
                  answer:{" "}
                  <span className="text-foreground">{world.answer}</span>
                </span>
              )}
              {world && world.gaveUp && (
                <span style={{ color: ACCENT }}>
                  ✗ agent stopped early
                </span>
              )}
            </div>
          </div>

          {/* The agent's action trace, the actions it has issued so far. */}
          <div className="flex flex-col gap-1.5 px-4 py-3">
            <span className="text-muted-foreground text-xs">
              agent actions ({state.cursor}/{totalActions})
            </span>
            <ol className="flex flex-col gap-1 text-xs">
              {resolved.actions.map((action, i) => {
                const done = i < state.cursor;
                const current = i === state.cursor - 1;
                return (
                  <li
                    key={`${action.label}-${i}`}
                    className="flex items-center gap-2"
                    style={{ opacity: done ? 1 : 0.35 }}
                  >
                    <span
                      aria-hidden
                      className="flex size-4 items-center justify-center rounded-full text-[10px] font-bold text-white"
                      style={{
                        backgroundColor: done
                          ? action.progress
                            ? SAGE
                            : ACCENT
                          : "var(--color-muted)",
                      }}
                    >
                      {done ? (action.progress ? "✓" : "!") : i + 1}
                    </span>
                    <code className={current ? "text-foreground" : undefined}>
                      {action.label}
                    </code>
                  </li>
                );
              })}
            </ol>
            {resolved.note && (
              <p className="text-muted-foreground mt-1 text-xs">
                ℹ {resolved.note}
              </p>
            )}
          </div>
        </div>

        {/* The scoreboard: both metrics, side by side, on the finished run. */}
        <div className="border-foreground/10 flex flex-wrap items-center gap-6 border-t px-4 py-3">
          <Light
            on={showResults}
            pass={reward.functional === 1}
            label={`Functional reward ${showResults ? reward.functional : "?"}`}
          />
          <Light
            on={showResults}
            pass={reward.surface === 1}
            label={`Surface-form match ${showResults ? reward.surface : "?"}`}
          />
          {showResults && reward.functional !== reward.surface && (
            <Badge variant="secondary">the two metrics disagree</Badge>
          )}
        </div>
      </div>

      <PlayPauseStepControls
        playing={playing}
        onPlayPause={onPlayPause}
        onStep={onStep}
        onReset={onReset}
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Set up the run">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Task</span>
            <div className="flex flex-wrap gap-1.5">
              {TASKS.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.title}
                  active={params.taskId === item.id}
                  onClick={() => intervene({ type: "setTask", id: item.id })}
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Agent plan</span>
            <div className="flex flex-wrap gap-1.5">
              {(["reference", "alternative", "offTarget"] as PlanName[]).map(
                (plan) => (
                  <Toggle
                    key={plan}
                    label={PLAN_LABELS[plan]}
                    active={params.plan === plan}
                    disabled={!task.achievable}
                    onClick={() => intervene({ type: "setPlan", plan })}
                  />
                ),
              )}
            </div>
            {!task.achievable && (
              <p className="text-muted-foreground text-xs">
                Plan is decided by the UA hint on an unachievable task.
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="UA hint"
              active={params.uaHint}
              onClick={() => intervene({ type: "toggleUaHint" })}
            />
          </div>
          <p className="text-muted-foreground text-xs">
            The UA hint tells the agent it may stop if a task looks impossible.
            It rescues unachievable tasks and trips feasible ones.
          </p>
          <div className="flex items-center gap-3">
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() =>
                intervene({
                  type: "setSeed",
                  value: (params.seed + 1) % SEED_COUNT,
                })
              }
            >
              Try another seed
            </Button>
            <span className="text-muted-foreground text-xs">
              Resamples whether the UA hint trips the agent on this task.
            </span>
          </div>
        </Panel>

        <Panel title="Scoring">
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Functional reward"
              active={metric === "functional"}
              onClick={() =>
                intervene({ type: "setMetric", metric: "functional" })
              }
            />
            <Toggle
              label="Surface-form"
              active={metric === "surface"}
              onClick={() =>
                intervene({ type: "setMetric", metric: "surface" })
              }
            />
          </div>
          {metric === "functional" ? (
            <div className="flex flex-col gap-1.5">
              <span className="text-muted-foreground text-xs">
                checks on the world state
              </span>
              {reward.checks.map((check) => (
                <div
                  key={check.label}
                  className="flex items-center gap-2 text-xs"
                >
                  <span
                    aria-hidden
                    className="flex size-4 items-center justify-center rounded-full text-[10px] font-bold text-white"
                    style={{
                      backgroundColor: !showResults
                        ? "var(--color-muted)"
                        : check.pass
                          ? SAGE
                          : ACCENT,
                    }}
                  >
                    {!showResults ? "•" : check.pass ? "✓" : "✗"}
                  </span>
                  <span
                    className={
                      check.pass && showResults
                        ? "text-foreground"
                        : "text-muted-foreground"
                    }
                  >
                    {check.label}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <span className="text-muted-foreground text-xs">
                action kinds vs the reference script
              </span>
              <table className="w-full text-left font-mono text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="py-1 font-medium">reference</th>
                    <th className="py-1 font-medium">agent</th>
                    <th className="py-1 text-right font-medium">match</th>
                  </tr>
                </thead>
                <tbody>
                  {reward.surfaceDiff.map((row, i) => (
                    <tr
                      key={`${row.ref}-${row.got}-${i}`}
                      className={
                        row.match ? "text-muted-foreground" : "text-foreground"
                      }
                    >
                      <td className="py-0.5">{row.ref}</td>
                      <td className="py-0.5">{row.got}</td>
                      <td className="py-0.5 text-right">
                        <span style={{ color: row.match ? SAGE : ACCENT }}>
                          {row.match ? "✓" : "✗"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="outline">
              {metric} score {showResults ? visibleScore : "pending"}
            </Badge>
            <Badge variant="outline">
              {state.cursor}/{totalActions} actions run
            </Badge>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            {/* Relative wrapper lets the bottom-fade mask sit flush against the scroll edge. */}
            <div className="relative">
              <div
                ref={logScrollRef}
                className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1"
              >
                {state.log.length === 0 ? (
                  <p className="text-muted-foreground px-2 py-1.5 text-xs">
                    Step the agent, change the plan, or flip the UA hint to
                    start the log.
                  </p>
                ) : (
                  <ul className="text-xs">
                    {[...state.log].reverse().map((entry) => (
                      <li key={entry.id}>
                        <button
                          type="button"
                          className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                          onClick={() =>
                            intervene({
                              type: "restore",
                              params: entry.snapshot,
                              label: `rewind to step ${entry.id}`,
                            })
                          }
                        >
                          {entry.id}. {entry.label}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {/* Fade mask signals that content continues below the visible area. */}
              {state.log.length > 3 && (
                <div
                  aria-hidden
                  className="from-background pointer-events-none absolute right-0 bottom-0 left-0 h-5 rounded-b-lg bg-gradient-to-t to-transparent"
                />
              )}
            </div>
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs 3 tiny hand-built tasks across a stub of WebArena&apos;s
        shopping, repo, and information sites, where the real benchmark runs 812
        tasks over four full self-hosted websites. The functional checks read
        the world state the actions wrote, exactly as the paper&apos;s reward
        functions do. Runs are deterministic: same preset and same seed produce
        the same trajectory. &ldquo;Try another seed&rdquo; resamples whether
        the UA hint trips the agent on the current feasible task. The 14.41%
        headline is GPT-4&apos;s end-to-end success rate, against 78.24% for
        humans.
      </p>
    </div>
  );
}
