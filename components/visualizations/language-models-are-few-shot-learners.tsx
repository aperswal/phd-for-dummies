"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  CONTEXT_BUDGET,
  getModelSize,
  getTask,
  initialState,
  MODEL_SIZES,
  predictSuccess,
  PRESETS,
  runningStats,
  step,
  TASKS,
  type Intervention,
  type PromptConfig,
  type SimState,
} from "@/components/visualizations/language-models-are-few-shot-learners-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#7a8a6f";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function reducer(state: SimState, action: ViewAction): SimState {
  return step(state, action);
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
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

// The predicted in-context learning curve for the current model and task, swept
// over every example count the prompt can hold, with the current point marked.
function LearningCurve({ config }: { config: PromptConfig }) {
  const task = getTask(config.taskId);
  const maxExamples = task.examples.length;
  const width = 460;
  const height = 230;
  const padL = 44;
  const padR = 16;
  const padT = 18;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const xFor = (examples: number) =>
    padL + (maxExamples === 0 ? 0 : (examples / maxExamples) * plotW);
  const yFor = (p: number) => padT + (1 - p) * plotH;

  const curves = MODEL_SIZES.map((size) => {
    const points = Array.from({ length: maxExamples + 1 }, (_, examples) => {
      const p = predictSuccess({
        ...config,
        modelId: size.id,
        exampleCount: examples,
      });
      return `${xFor(examples)},${yFor(p)}`;
    }).join(" ");
    const isCurrent = size.id === config.modelId;
    return { size, points, isCurrent };
  });

  const current = predictSuccess(config);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`In-context learning curves. The ${
        getModelSize(config.modelId).label
      } model reaches ${percent(current)} predicted success with ${
        config.exampleCount
      } examples in the prompt.`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={`grid-${p}`}>
          <line
            x1={padL}
            x2={width - padR}
            y1={yFor(p)}
            y2={yFor(p)}
            className="stroke-foreground/10"
            strokeWidth={1}
          />
          <text
            x={padL - 6}
            y={yFor(p) + 3}
            textAnchor="end"
            className="fill-muted-foreground"
            style={{ fontSize: 9 }}
          >
            {percent(p)}
          </text>
        </g>
      ))}

      {curves.map(({ size, points, isCurrent }) => (
        <polyline
          key={size.id}
          points={points}
          fill="none"
          stroke={isCurrent ? ACCENT : SAGE}
          strokeWidth={isCurrent ? 2.6 : 1}
          strokeOpacity={isCurrent ? 1 : 0.35}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}

      <circle
        cx={xFor(config.exampleCount)}
        cy={yFor(current)}
        r={5}
        fill={ACCENT}
      />
      <line
        x1={xFor(config.exampleCount)}
        x2={xFor(config.exampleCount)}
        y1={yFor(current)}
        y2={padT + plotH}
        stroke={ACCENT}
        strokeWidth={1}
        strokeDasharray="3 3"
        strokeOpacity={0.5}
      />

      {Array.from({ length: maxExamples + 1 }, (_, examples) => (
        <text
          key={`x-${examples}`}
          x={xFor(examples)}
          y={height - 18}
          textAnchor="middle"
          className="fill-muted-foreground"
          style={{ fontSize: 9 }}
        >
          {examples}
        </text>
      ))}
      <text
        x={padL + plotW / 2}
        y={height - 4}
        textAnchor="middle"
        className="fill-muted-foreground"
        style={{ fontSize: 10 }}
      >
        examples in the prompt
      </text>
    </svg>
  );
}

// The prompt itself, drawn as a recipe card: the task description, a stack of
// worked example lines (some mislabeled), and the blank line the model fills.
function PromptCard({ config }: { config: PromptConfig }) {
  const task = getTask(config.taskId);
  const shown = task.examples.slice(0, config.exampleCount);
  return (
    <div className="bg-foreground/[0.03] ring-foreground/10 flex flex-col gap-1.5 rounded-lg p-3 font-mono text-xs ring-1">
      {config.useDescription ? (
        <div className="text-foreground">{task.description}</div>
      ) : (
        <div className="text-muted-foreground/50 line-through">
          {task.description}
        </div>
      )}
      {shown.map((example, index) => {
        const corrupted =
          index >= config.exampleCount - config.corruptedExamples;
        return (
          <div
            key={`ex-${index}`}
            className="flex items-center justify-between gap-2"
            style={{ color: corrupted ? ACCENT : SAGE }}
          >
            <span>
              {example.input} {"=>"}{" "}
              {corrupted ? scrambleOutput(example.output) : example.output}
            </span>
            {corrupted && (
              <span className="shrink-0 text-[10px] tracking-wide uppercase">
                mislabeled
              </span>
            )}
          </div>
        );
      })}
      {shown.length === 0 && (
        <div className="text-muted-foreground/60">(no examples)</div>
      )}
      <div className="text-foreground flex items-center gap-2">
        <span>{task.examples[0]?.input ?? "?"}</span>
        <span>{"=>"}</span>
        <span
          className="inline-block h-3 w-10 animate-pulse rounded"
          style={{ backgroundColor: ACCENT_SOFT }}
          aria-label="the answer the model writes"
        />
      </div>
    </div>
  );
}

function scrambleOutput(output: string): string {
  if (/^\d+$/.test(output)) return String(Number(output) + 1);
  return output.slice(1) + output.slice(0, 1);
}

export function LanguageModelsAreFewShotLearners() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("few-shot-175b"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [presetId, setPresetId] = useState("few-shot-175b");

  const { config } = state;
  const task = getTask(config.taskId);
  const stats = useMemo(() => runningStats(state), [state]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 420 / speed, () =>
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
    intervene({ type: "loadPreset", id: presetId });
  }, [intervene, presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      intervene({ type: "loadPreset", id });
    },
    [intervene, resetClock],
  );

  const budgetFraction = Math.min(stats.tokensUsed / CONTEXT_BUDGET, 1.15);

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

      <div className="grid gap-4 lg:grid-cols-2">
        <SimPanel title="The prompt">
          <PromptCard config={config} />
          <div className="flex flex-wrap gap-1.5">
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => intervene({ type: "addExample" })}
              disabled={config.exampleCount >= task.examples.length}
            >
              + example
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => intervene({ type: "removeExample" })}
              disabled={config.exampleCount <= 0}
            >
              - example
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => intervene({ type: "corruptExample" })}
              disabled={config.corruptedExamples >= config.exampleCount}
            >
              mislabel one
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => intervene({ type: "fixExamples" })}
              disabled={config.corruptedExamples === 0}
            >
              fix all
            </Button>
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-muted-foreground flex items-center justify-between text-xs">
              <span>context window</span>
              <span
                className="font-mono"
                style={{ color: stats.overBudget ? ACCENT : undefined }}
              >
                {stats.tokensUsed} / {CONTEXT_BUDGET} tokens
              </span>
            </div>
            <div className="bg-foreground/10 h-2 w-full overflow-hidden rounded">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.min(budgetFraction, 1) * 100}%`,
                  backgroundColor: stats.overBudget ? ACCENT : SAGE,
                }}
              />
            </div>
            {stats.overBudget && (
              <span className="text-xs" style={{ color: ACCENT }}>
                The prompt overflows the window. The earliest examples fall out.
              </span>
            )}
          </div>
        </SimPanel>

        <SimPanel title="In-context learning curve">
          <LearningCurve config={config} />
          <p className="text-muted-foreground text-xs">
            Orange is the {getModelSize(config.modelId).label} model you picked.
            The faint sage lines are the other sizes. The slope is what changes
            with scale.
          </p>
        </SimPanel>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
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
            max={4}
            step={0.5}
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
            className="h-1.5 w-full cursor-pointer"
            style={{ accentColor: ACCENT }}
            aria-label="Playback speed"
          />
        </label>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Model size</span>
            <div className="flex flex-wrap gap-1.5">
              {MODEL_SIZES.map((size) => (
                <Toggle
                  key={size.id}
                  label={size.label}
                  active={config.modelId === size.id}
                  onClick={() => intervene({ type: "setModel", id: size.id })}
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">Task</span>
            <div className="flex flex-wrap gap-1.5">
              {TASKS.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.label}
                  active={config.taskId === item.id}
                  onClick={() => intervene({ type: "setTask", id: item.id })}
                />
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Task description"
              active={config.useDescription}
              onClick={() => intervene({ type: "toggleDescription" })}
            />
          </div>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground flex items-center justify-between">
              <span>Seed</span>
              <span className="text-foreground font-mono">{config.seed}</span>
            </span>
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={config.seed}
              onChange={(event) =>
                intervene({
                  type: "setSeed",
                  value: Number(event.target.value),
                })
              }
              className="h-1.5 w-full cursor-pointer"
              style={{ accentColor: ACCENT }}
              aria-label="Random seed"
            />
          </label>
          <p className="text-muted-foreground text-xs">{task.note}</p>
        </SimPanel>

        <SimPanel title="Running the test queries">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              predicted {percent(stats.predicted)}
            </Badge>
            <Badge variant="outline">
              accuracy {stats.attempted > 0 ? percent(stats.accuracy) : "-"}
            </Badge>
            <Badge variant="outline">
              {stats.correct}/{stats.attempted} correct
            </Badge>
          </div>
          <div className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">query</th>
                  <th className="px-2 py-1 font-medium">model</th>
                  <th className="px-2 py-1 text-right font-medium">answer</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {state.trials.length === 0 ? (
                  <tr>
                    <td className="text-muted-foreground px-2 py-2" colSpan={3}>
                      Press play to send test queries through the prompt.
                    </td>
                  </tr>
                ) : (
                  [...state.trials].reverse().map((trial) => (
                    <tr
                      key={trial.id}
                      className={
                        trial.correct
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }
                    >
                      <td className="px-2 py-1">{trial.prompt}</td>
                      <td className="px-2 py-1">{trial.guess}</td>
                      <td
                        className="px-2 py-1 text-right"
                        style={{ color: trial.correct ? SAGE : ACCENT }}
                      >
                        {trial.correct ? "correct" : `want ${trial.truth}`}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Add an example, scale the model, or mislabel a demo to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  config: event.snapshot,
                  label: `rewind to step ${event.id}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The success rates here are fit to the paper&apos;s own numbers, GPT-3
        175B going from 76.9% to 100% on 2-digit addition (Table 3.9) and random
        insertion climbing 8% to 67% (Table 3.10), while reversed words stays
        near zero. Each test query is a seeded coin flip against the predicted
        rate, so the running accuracy lands on the curve over many trials, the
        way the paper reports accuracy over 2,000 instances.
      </p>
    </div>
  );
}
