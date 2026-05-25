"use client";

import { useCallback, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  DEFAULT_BUDGET,
  getTask,
  groundChance,
  initialState,
  OBSERVATION_MODES,
  PERTURBATION_LABEL,
  PRESETS,
  step,
  TASKS,
  type Attempt,
  type Intervention,
  type Perturbation,
  type SimState,
} from "@/components/visualizations/osworld-model";

// ACCENT (clay-orange) = the warning/problem signal: missed clicks, gauge fill.
// SAGE (sage green)    = progress/success: completed subgoals, landed clicks.
// SLATE                = neutral secondary: popup-clearing steps.
// Active subgoal uses a CSS foreground outline, not ACCENT, so "in progress"
// never reads as "broken".
const ACCENT = "#c2683f";
const SAGE = "#7c8a6b";
const SLATE = "#6b7785";

const PERTURBATIONS: Perturbation[] = ["none", "move", "shrink", "clutter"];

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

const OUTCOME_COLOR: Record<Attempt["outcome"], string> = {
  land: SAGE,
  miss: ACCENT,
  popup: SLATE,
};

const OUTCOME_LABEL: Record<Attempt["outcome"], string> = {
  land: "landed",
  miss: "missed",
  popup: "popup",
};

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

const PHASE_TEXT: Record<SimState["phase"], string> = {
  ready: "ready",
  running: "running",
  done: "done, reward 1",
  failed: "failed, reward 0",
};

export function OSWorldAgentLoop() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("single-app"),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState("single-app");

  const {
    params,
    attempts,
    phase,
    subgoalIndex,
    step: stepCount,
    reward,
    log,
  } = state;
  const task = getTask(params.taskId);
  const chance = groundChance(params);
  const finished = phase === "done" || phase === "failed";

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Stop the clock once the run reaches a terminal state, so play doesn't spin
  // on a finished task.
  const resetClock = useFixedTimestep(playing && !finished, 900, () =>
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

  // SVG layout. A column of subgoal tiles on the left tracks the plan; a tape of
  // attempt dots runs along the bottom; a big landing-chance gauge sits on the
  // right because that single number drives the whole outcome.
  const width = 1000;
  const height = 360;
  const planX = 30;
  const planW = 360;
  const rowH = 44;
  const planTop = 70;
  const gaugeCx = 800;
  const gaugeCy = 150;
  const gaugeR = 78;
  const tapeY = 320;
  const tapeX0 = 30;
  const tapeMax = params.stepBudget;
  const tapeStep = (width - 60) / tapeMax;

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
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          role="img"
          aria-label={`Agent running '${task.label}'. ${subgoalIndex} of ${task.subgoals.length} subgoals done, step ${stepCount} of ${params.stepBudget}, landing chance ${percent(chance)}, state ${PHASE_TEXT[phase]}.`}
        >
          <text
            x={planX}
            y={48}
            className="fill-muted-foreground"
            style={{ fontSize: 13 }}
          >
            the plan ({task.app})
          </text>
          {task.subgoals.map((subgoal, index) => {
            const done = index < subgoalIndex;
            const active = index === subgoalIndex && !finished;
            const y = planTop + index * rowH;
            return (
              <g key={`goal-${index}`}>
                <rect
                  x={planX}
                  y={y}
                  width={planW}
                  height={rowH - 8}
                  rx={7}
                  fill={done ? SAGE : "var(--color-muted)"}
                  fillOpacity={done ? 1 : active ? 1 : 0.45}
                  stroke={active ? "var(--color-foreground)" : "transparent"}
                  strokeWidth={2}
                />
                <text
                  x={planX + 14}
                  y={y + (rowH - 8) / 2 + 4}
                  className="fill-foreground"
                  fill={done ? "#fff" : undefined}
                  fillOpacity={done || active ? 1 : 0.6}
                  style={{ fontSize: 13 }}
                >
                  {done ? "✓ " : active ? "→ " : ""}
                  {subgoal}
                </text>
              </g>
            );
          })}

          <circle
            cx={gaugeCx}
            cy={gaugeCy}
            r={gaugeR}
            fill="none"
            className="stroke-foreground/10"
            strokeWidth={14}
          />
          <circle
            cx={gaugeCx}
            cy={gaugeCy}
            r={gaugeR}
            fill="none"
            stroke={ACCENT}
            strokeWidth={14}
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * gaugeR}`}
            strokeDashoffset={`${2 * Math.PI * gaugeR * (1 - chance)}`}
            transform={`rotate(-90 ${gaugeCx} ${gaugeCy})`}
          />
          <text
            x={gaugeCx}
            y={gaugeCy - 4}
            textAnchor="middle"
            className="fill-foreground"
            style={{ fontSize: 30, fontWeight: 600 }}
          >
            {percent(chance)}
          </text>
          <text
            x={gaugeCx}
            y={gaugeCy + 18}
            textAnchor="middle"
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            click lands
          </text>
          <text
            x={gaugeCx}
            y={gaugeCy + gaugeR + 26}
            textAnchor="middle"
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            {
              OBSERVATION_MODES.find((m) => m.id === params.observationId)
                ?.label
            }
            {params.perturbation !== "none"
              ? ` + ${PERTURBATION_LABEL[params.perturbation].toLowerCase()}`
              : ""}
          </text>

          <line
            x1={tapeX0}
            y1={tapeY}
            x2={width - 30}
            y2={tapeY}
            className="stroke-foreground/15"
            strokeWidth={2}
          />
          {Array.from({ length: tapeMax }, (_, i) => (
            <circle
              key={`slot-${i}`}
              cx={tapeX0 + (i + 0.5) * tapeStep}
              cy={tapeY}
              r={3}
              className="fill-foreground/15"
            />
          ))}
          {attempts.map((attempt) => (
            <circle
              key={`a-${attempt.step}`}
              cx={tapeX0 + (attempt.step - 0.5) * tapeStep}
              cy={tapeY}
              r={7}
              fill={OUTCOME_COLOR[attempt.outcome]}
            >
              <title>
                step {attempt.step}: {OUTCOME_LABEL[attempt.outcome]} &apos;
                {attempt.subgoal}&apos;
              </title>
            </circle>
          ))}
          <text
            x={tapeX0}
            y={tapeY - 14}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            trajectory: each dot is one action, of {params.stepBudget} allowed
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={finished}
        />
        <Badge variant={reward === 1 ? "default" : "secondary"}>
          {PHASE_TEXT[phase]}
        </Badge>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="The task">
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
          <span className="text-muted-foreground text-xs font-medium">
            How the agent sees — observation mode
          </span>
          <div className="flex flex-wrap gap-1.5">
            {OBSERVATION_MODES.map((mode) => (
              <Toggle
                key={mode.id}
                label={mode.label}
                active={params.observationId === mode.id}
                onClick={() =>
                  intervene({ type: "setObservation", id: mode.id })
                }
              />
            ))}
          </div>
          <span className="text-muted-foreground text-xs font-medium">
            Window state — perturb mid-run to watch the gauge drop
          </span>
          <div className="flex flex-wrap gap-1.5">
            {PERTURBATIONS.map((value) => (
              <Toggle
                key={value}
                label={PERTURBATION_LABEL[value]}
                active={params.perturbation === value}
                onClick={() => intervene({ type: "setPerturbation", value })}
              />
            ))}
          </div>
          <Slider
            label="Step budget — how many actions before the run fails"
            value={params.stepBudget}
            min={5}
            max={30}
            step={1}
            display={String(params.stepBudget)}
            onChange={(value) => intervene({ type: "setBudget", value })}
          />
          {DEFAULT_BUDGET !== params.stepBudget && (
            <p className="text-muted-foreground text-xs">
              Default budget is {DEFAULT_BUDGET} steps, matching the paper.
            </p>
          )}
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => intervene({ type: "reshuffle" })}
          >
            New sample — try a different trajectory
          </Button>
        </SimPanel>

        <SimPanel title="Inside the run">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              {subgoalIndex}/{task.subgoals.length} subgoals
            </Badge>
            <Badge variant="outline">
              step {stepCount}/{params.stepBudget}
            </Badge>
            <Badge variant="outline">click lands {percent(chance)}</Badge>
            <Badge variant="outline">
              misses {attempts.filter((a) => a.outcome !== "land").length}
            </Badge>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step or play to start the agent loop."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The plan here is always right, because the paper&apos;s agents plan well
        and fail at grounding, so the only question each step is whether the
        click lands. The landing chance comes from the paper: the observation
        mode sets the base rate near Table 5&apos;s overall numbers (a11y tree
        beats raw screenshot), and each window perturbation scales it by the
        drop Figure 8 measured (original 50.79%, moved 36.65%, shrunk 15.04%,
        cluttered 25.39%). A real agent&apos;s accuracy comes from the model;
        here it is a fixed rate so the loop is legible. Each preset uses a fixed
        seed so the run is deterministic; press &ldquo;New sample&rdquo; to try
        a different trajectory.
      </p>
    </div>
  );
}
