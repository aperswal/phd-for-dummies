"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  getMode,
  getProblem,
  initialState,
  MODES,
  PRESETS,
  PROBLEMS,
  solve,
  step,
  stepErrorRate,
  type Intervention,
  type SimState,
} from "@/components/visualizations/chain-of-thought-prompting-model";

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

// Scale slider runs over a log axis from 0.4B to 540B, the model sizes in the
// paper, so the emergence threshold near 100B sits where the reader can find it.
const SCALE_STOPS = [0.4, 1.3, 6.7, 8, 62, 137, 175, 540];

function nearestScale(raw: number): number {
  return SCALE_STOPS.reduce((best, stop) =>
    Math.abs(stop - raw) < Math.abs(best - raw) ? stop : best,
  );
}

export function ChainOfThought() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("standard-fails"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("standard-fails");

  const { params, revealed, log } = state;
  const solution = useMemo(() => solve(params), [params]);
  const problem = getProblem(params.problemId);
  const mode = getMode(params.mode);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 900 / speed, () =>
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

  const totalSteps = solution.steps.length;
  const shown = mode.emitsSteps ? Math.min(revealed, totalSteps) : totalSteps;
  const chainDone = !mode.emitsSteps || shown >= totalSteps;
  const errorRate = mode.reasoningBefore
    ? stepErrorRate(params.scaleB)
    : stepErrorRate(100 / 12);

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
        <div className="border-foreground/10 border-b p-4">
          <span className="text-muted-foreground text-xs font-medium">
            Question
          </span>
          <p className="text-foreground mt-1 text-sm">{problem.question}</p>
        </div>

        <div className="flex flex-col gap-2 p-4">
          {!mode.emitsSteps && (
            <div className="flex flex-col gap-2">
              {mode.id === "dots" && (
                <p className="text-muted-foreground font-mono text-sm">
                  {".".repeat(Math.max(solution.dots, 1))}
                </p>
              )}
              <p className="text-muted-foreground text-sm">
                No reasoning steps. The model commits to one answer.
              </p>
            </div>
          )}

          {mode.emitsSteps &&
            solution.steps.map((emitted, index) => {
              const visible = index < shown;
              const broken = emitted.faulted;
              return (
                <button
                  key={`step-${emitted.index}`}
                  type="button"
                  aria-pressed={params.injected.includes(emitted.index)}
                  aria-label={`Toggle a fault on step ${emitted.index + 1}`}
                  onClick={() =>
                    intervene({ type: "toggleInjected", index: emitted.index })
                  }
                  className="hover:bg-foreground/5 flex items-start gap-3 rounded-lg p-2 text-left ring-1 transition-opacity"
                  style={{
                    opacity: visible ? 1 : 0.2,
                    boxShadow:
                      broken && visible
                        ? `inset 0 0 0 1px ${ACCENT}`
                        : undefined,
                  }}
                >
                  <span
                    className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium"
                    style={{
                      background:
                        broken && visible ? ACCENT : "var(--color-muted)",
                      color: broken && visible ? "#fff" : undefined,
                    }}
                  >
                    {emitted.index + 1}
                  </span>
                  <span className="flex flex-col gap-0.5 text-sm">
                    <span className="text-foreground">{emitted.text}</span>
                    {visible && broken && (
                      <span className="text-xs" style={{ color: ACCENT }}>
                        wrong step (should be {emitted.trueValue})
                        {emitted.reason === "injected"
                          ? ", you broke it"
                          : ", the model slipped at this scale"}
                      </span>
                    )}
                  </span>
                </button>
              );
            })}

          {chainDone && (
            <div className="mt-1 flex items-center gap-2 rounded-lg p-2 text-sm">
              <span className="text-foreground font-medium">
                The answer is {solution.finalAnswer} {problem.unit}.
              </span>
              <Badge
                variant="outline"
                style={{
                  borderColor: solution.correct ? SAGE : ACCENT,
                  color: solution.correct ? SAGE : ACCENT,
                }}
              >
                {solution.correct
                  ? "correct"
                  : `wrong, true answer ${solution.trueAnswer}`}
              </Badge>
            </div>
          )}
        </div>
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
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs">Prompt mode</span>
            <div className="flex flex-wrap gap-1.5">
              {MODES.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.label}
                  active={params.mode === item.id}
                  onClick={() => intervene({ type: "setMode", mode: item.id })}
                />
              ))}
            </div>
            <p className="text-muted-foreground text-xs">{mode.blurb}</p>
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs">Problem</span>
            <div className="flex flex-wrap gap-1.5">
              {PROBLEMS.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.label}
                  active={params.problemId === item.id}
                  onClick={() => intervene({ type: "setProblem", id: item.id })}
                />
              ))}
            </div>
          </div>

          <Slider
            label="Model scale (billions of parameters)"
            value={params.scaleB}
            min={0.4}
            max={540}
            step={0.1}
            display={`${params.scaleB}B`}
            onChange={(value) =>
              intervene({ type: "setScale", value: nearestScale(value) })
            }
          />
          <Slider
            label="Seed (reshuffles where the model slips)"
            value={params.seed}
            min={0}
            max={12}
            step={1}
            display={String(params.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the run">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{mode.label}</Badge>
            <Badge variant="outline">scale {params.scaleB}B</Badge>
            <Badge variant="outline">
              per-step error {Math.round(errorRate * 100)}%
            </Badge>
            <Badge variant="outline">
              {mode.emitsSteps
                ? `${solution.steps.filter((s) => s.faulted).length} broken steps`
                : "no steps"}
            </Badge>
          </div>
          <p className="text-muted-foreground text-xs">
            {mode.reasoningBefore
              ? params.scaleB >= 100
                ? "Big enough to reason. Each step lands, so the chain reaches the right answer."
                : "Below the emergence scale near 100B. The steps read fluently but a value slips, and the slip flows downstream."
              : "This mode writes no natural-language reasoning before the answer, so a bigger model buys it nothing. It scores like the baseline."}
          </p>
          {mode.emitsSteps && (
            <p className="text-muted-foreground text-xs">
              Click any step to break it by hand and watch the wrong value flow
              into every step after it.
            </p>
          )}
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Flip a mode, drag scale, or break a step to start the log."
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
        The reasoning here is hand-set so each step has a known right value and
        a broken step poisons the ones after it, the way the paper&apos;s error
        analysis describes. A real model writes the words itself, and the
        emergence curve is sharper than this smooth stand-in. This runs{" "}
        {PROBLEMS.length} short problems, where the paper tests thousands across
        five math, commonsense, and symbolic benchmarks.
      </p>
    </div>
  );
}
