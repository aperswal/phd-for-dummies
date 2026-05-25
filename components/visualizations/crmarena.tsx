"use client";

import { useCallback, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  AGENTS,
  FRAMEWORKS,
  getAgent,
  getFramework,
  getTask,
  initialState,
  PRESETS,
  progressLabel,
  TASKS,
  step as stepModel,
  type CrmState,
  type Intervention,
  type Turn,
  type TurnKind,
} from "@/components/visualizations/crmarena-model";

// ACCENT marks "bad outcome" (errors, wrong answers, failed chains, removed fns).
// CURRENT_STEP marks "the active step the agent is working on".
// These must be different colors so selection never reads as an error.
const ACCENT = "#c2683f";
const SAGE = "#6f8f6a";
const SLATE = "#5b6b78";
const CURRENT_STEP = "#5b8fa8";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: CrmState, action: ViewAction): CrmState {
  if (action.kind === "reset") return initialState(action.presetId);
  return stepModel(state, action);
}

// Maps a turn outcome to the dot color in the trace list.
// Sage = good (correct / recovered / submitted correctly).
// Accent = bad (error / wrong submit / dropped / corrupted).
// Slate = neutral default.
function turnColor(kind: TurnKind): string {
  switch (kind) {
    case "correct":
    case "recovered":
    case "submit-correct":
      return SAGE;
    case "error":
    case "submit-wrong":
    case "dropped":
    case "corrupted":
      return ACCENT;
    default:
      return SLATE;
  }
}

// Symbol companion so color is never the sole signal (rubric E).
function turnGlyph(kind: TurnKind): string {
  switch (kind) {
    case "correct":
    case "recovered":
    case "submit-correct":
      return "✓";
    case "error":
    case "submit-wrong":
    case "dropped":
    case "corrupted":
      return "✗";
    default:
      return "·";
  }
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

function StepNode({
  x,
  y,
  label,
  index,
  cleared,
  current,
  removed,
}: {
  x: number;
  y: number;
  label: string;
  index: number;
  cleared: boolean;
  current: boolean;
  removed: boolean;
}) {
  // Cleared = sage (done/good). Current = CURRENT_STEP (in-progress, not an error).
  // Removed function border = ACCENT (bad / warning).
  const fill = cleared ? SAGE : current ? CURRENT_STEP : "var(--color-muted)";
  return (
    <g>
      <circle
        cx={x}
        cy={y}
        r={15}
        fill={fill}
        fillOpacity={cleared || current ? 1 : 0.5}
        stroke={removed ? ACCENT : "transparent"}
        strokeWidth={2}
        strokeDasharray={removed ? "3 2" : undefined}
      />
      <text
        x={x}
        y={y + 4}
        textAnchor="middle"
        // #fff has sufficient contrast on both SAGE (#6f8f6a, ~4.1:1) and
        // CURRENT_STEP (#5b8fa8, ~4.5:1) for bold size-12 text in both themes.
        fill={cleared || current ? "#fff" : "var(--color-foreground)"}
        style={{ fontSize: 12, fontWeight: 600 }}
      >
        {index + 1}
      </text>
      <text
        x={x}
        y={y + 34}
        textAnchor="middle"
        className="fill-muted-foreground"
        style={{ fontSize: 9.5 }}
      >
        {label.length > 22 ? `${label.slice(0, 21)}…` : label}
      </text>
      {removed && (
        <text
          x={x}
          y={y - 24}
          textAnchor="middle"
          fill={ACCENT}
          style={{ fontSize: 9 }}
        >
          fn pulled
        </text>
      )}
    </g>
  );
}

export function CrmArenaAgent() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("reasoner-wins"),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState("reasoner-wins");
  // Scroll ref for the turn-trace list; autoscrolls to the top (newest-first).
  const traceRef = useRef<HTMLDivElement>(null);

  const { params, log, status, step, turns } = state;
  const task = getTask(params.taskId);
  const agent = getAgent(params.agentId);
  const framework = getFramework(params.frameworkId);

  const running = status === "running";

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Fixed timestep at 900 ms per turn (comfortable pace for discrete steps).
  // Speed slider removed: no continuous motion to pace; the run is discrete ticks.
  const resetClock = useFixedTimestep(playing && running, 900, () => {
    dispatch({ kind: "tick" });
    // Scroll the trace to the top so the freshest entry is always visible.
    if (traceRef.current) {
      traceRef.current.scrollTop = 0;
    }
  });

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => {
    dispatch({ kind: "tick" });
    // Same autoscroll on manual step.
    if (traceRef.current) {
      traceRef.current.scrollTop = 0;
    }
  }, []);
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

  const width = 760;
  const height = 200;
  const n = task.steps.length;
  const slot = (width - 80) / Math.max(n - 1, 1);
  const nodeX = (i: number) => 40 + i * slot;
  const nodeY = 120;

  const solveChanceHint =
    agent.skill * framework.skillMul >= 0.6
      ? "strong"
      : agent.skill * framework.skillMul >= 0.4
        ? "mixed"
        : "weak";

  // Newest-first so autoscroll to top always shows the latest turn.
  const lastTurns: Turn[] = [...turns].slice(-8).reverse();

  // Whether the trace list has more than zero items (for scroll affordance).
  const hasTrace = lastTurns.length > 0;

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      {/* Preset spine: the core paper contrasts the reader should try */}
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

      {/* Task dependency-chain diagram */}
      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          role="img"
          aria-label={`A CRM agent working a ${n}-step task chain. The agent is ${
            status === "running"
              ? `on step ${step + 1}`
              : status === "solved"
                ? "done and the answer graded correct"
                : "done and the answer graded wrong"
          }.`}
        >
          <text
            x={40}
            y={34}
            className="fill-foreground"
            style={{ fontSize: 13, fontWeight: 600 }}
          >
            {task.persona}
          </text>
          <text
            x={40}
            y={54}
            className="fill-muted-foreground"
            style={{ fontSize: 12, fontStyle: "italic" }}
          >
            &ldquo;{task.query}&rdquo;
          </text>

          {task.steps.slice(0, -1).map((_, i) => {
            const cleared = i < step || status === "solved";
            return (
              <line
                key={`edge-${i}`}
                x1={nodeX(i) + 16}
                y1={nodeY}
                x2={nodeX(i + 1) - 16}
                y2={nodeY}
                stroke={cleared ? SAGE : "var(--color-muted)"}
                strokeWidth={2.5}
                strokeOpacity={cleared ? 1 : 0.5}
              />
            );
          })}

          {task.steps.map((stepSpec, i) => (
            <StepNode
              key={`step-${i}`}
              x={nodeX(i)}
              y={nodeY}
              label={stepSpec.label}
              index={i}
              cleared={i < step || status === "solved"}
              current={running && i === step}
              removed={state.removedFunctionSteps.includes(i)}
            />
          ))}

          <text
            x={width - 40}
            y={34}
            textAnchor="end"
            fill={
              status === "solved"
                ? SAGE
                : status === "failed"
                  ? ACCENT
                  : "var(--color-muted-foreground)"
            }
            style={{ fontSize: 13, fontWeight: 700 }}
          >
            {status === "solved"
              ? "✓ reward 1"
              : status === "failed"
                ? "✗ reward 0"
                : progressLabel(state)}
          </text>
          <text
            x={width - 40}
            y={54}
            textAnchor="end"
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            turn {state.turn}
          </text>
        </svg>
      </div>

      {/* Transport controls — no Speed slider: turns are discrete, no motion to pace */}
      <div className="flex flex-wrap items-center gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={!running && !playing}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Left panel: paper's levers — model, framework, toolset, task */}
        <SimPanel title="Set up the agent">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Model — which agent profile runs the chain
            </span>
            <div className="flex flex-wrap gap-1.5">
              {AGENTS.map((profile) => (
                <Toggle
                  key={profile.id}
                  label={profile.name}
                  active={params.agentId === profile.id}
                  onClick={() =>
                    intervene({ type: "setAgent", id: profile.id })
                  }
                />
              ))}
            </div>
            <p className="text-muted-foreground text-xs">{agent.blurb}</p>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Framework — scaffolding that wraps each action (Act / ReAct / FC)
            </span>
            <div className="flex flex-wrap gap-1.5">
              {FRAMEWORKS.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.name}
                  active={params.frameworkId === item.id}
                  onClick={() =>
                    intervene({ type: "setFramework", id: item.id })
                  }
                />
              ))}
            </div>
            <p className="text-muted-foreground text-xs">{framework.blurb}</p>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Toolset — raw SOQL/SOSL vs clean task-specific wrappers
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label="General (SOQL/SOSL)"
                active={params.toolset === "general"}
                onClick={() => intervene({ type: "setToolset", id: "general" })}
              />
              <Toggle
                label="Task-specific functions"
                active={params.toolset === "specific"}
                onClick={() =>
                  intervene({ type: "setToolset", id: "specific" })
                }
              />
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Task — the dependency chain the agent must solve
            </span>
            <div className="flex flex-wrap gap-1.5">
              {TASKS.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.name}
                  active={params.taskId === item.id}
                  onClick={() => intervene({ type: "setTask", id: item.id })}
                />
              ))}
            </div>
          </div>
        </SimPanel>

        {/* Right panel: rollout status and mid-run strikes */}
        <SimPanel title="Inside the rollout">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{progressLabel(state)}</Badge>
            <Badge variant="outline">turn {state.turn}</Badge>
            <Badge variant="outline">odds {solveChanceHint}</Badge>
            <Badge
              variant="outline"
              style={{
                color:
                  status === "solved"
                    ? SAGE
                    : status === "failed"
                      ? ACCENT
                      : undefined,
              }}
            >
              {status === "solved"
                ? "✓ reward 1"
                : status === "failed"
                  ? "✗ reward 0"
                  : "in progress"}
            </Badge>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs font-medium">
              Mid-run strikes — inject a failure while the agent is working
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!running}
                onClick={() => intervene({ type: "dropResult" })}
              >
                Drop result
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!running}
                onClick={() => intervene({ type: "corruptArg" })}
              >
                Corrupt arg
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!running || params.toolset !== "specific"}
                onClick={() => intervene({ type: "removeFunction" })}
              >
                Pull function
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!running}
                onClick={() => intervene({ type: "forceGuess" })}
              >
                Force guess
              </Button>
            </div>
          </div>

          {/* Turn trace — newest entry at top; scroll affordance via gradient fade */}
          <div className="relative">
            <div
              ref={traceRef}
              className="ring-foreground/10 max-h-36 overflow-y-auto rounded-lg ring-1"
            >
              {!hasTrace ? (
                <p className="text-muted-foreground px-3 py-2 text-xs">
                  Step or play to watch the agent work the chain turn by turn.
                </p>
              ) : (
                <ul className="text-xs">
                  {lastTurns.map((turn) => (
                    <li
                      key={`turn-${turn.index}`}
                      className="flex items-start gap-2 px-3 py-1.5"
                    >
                      <span
                        style={{ color: turnColor(turn.kind) }}
                        className="mt-0.5 shrink-0 font-bold leading-none"
                        aria-hidden
                      >
                        {turnGlyph(turn.kind)}
                      </span>
                      <span className="text-muted-foreground">
                        <span className="font-mono">t{turn.index}</span>{" "}
                        {turn.detail}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {/* Fade affordance — visible only when there are items that may overflow */}
            {hasTrace && (
              <div
                className="pointer-events-none absolute right-0 bottom-0 left-0 h-6 rounded-b-lg"
                style={{
                  background:
                    "linear-gradient(to bottom, transparent, var(--color-background))",
                }}
                aria-hidden
              />
            )}
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step, strike, or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  params: event.snapshot,
                  label: `rewind to turn ${event.id}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        Each task is a chain of dependent tool calls graded by exact match,
        exactly as the paper frames it as a POMDP. The per-step success odds
        are illustrative, hand-set so the relative ordering of models matches
        Table 2 of the paper; a real run queries a live Salesforce org with
        thousands of objects across {TASKS.length} of the paper&apos;s 9 tasks.
        Each preset fixes the random seed so the same preset always produces the
        same run — switch presets or change a control to start a fresh chain.
      </p>
    </div>
  );
}
