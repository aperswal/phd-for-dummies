"use client";

import { useCallback, useReducer, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  ALGORITHMS,
  computeAdvantages,
  getAlgorithm,
  getPreset,
  initialFromPreset,
  PRESETS,
  step,
  turn1HasSignal,
  turn1Spread,
  type Input,
  type Intervention,
  type LogEntry,
  type RewardWeights,
  type SimState,
} from "@/components/visualizations/reinforcing-multi-turn-reasoning-in-llm-agents-model";

const ACCENT = "#c2683f";
const SAGE = "#8a9b6e";
const NEGATIVE = "#5b6b7a";

const MAX_STEPS_PER_FRAME = 60;

function reducer(
  state: SimState,
  input: Input | { kind: "reset"; presetId: string },
): SimState {
  if (input.kind === "reset") return initialFromPreset(input.presetId);
  return step(state, input);
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

function EventLog({
  events,
  onRestore,
}: {
  events: LogEntry[];
  onRestore: (entry: LogEntry) => void;
}) {
  // Events are displayed newest-first so new entries appear at the top without
  // needing autoscroll. The fade gradient signals more entries exist below.
  return (
    <div className="relative">
      <div className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1">
        {events.length === 0 ? (
          <p className="text-muted-foreground px-2 py-1.5 text-xs">
            Switch the algorithm, drag a weight, or resample to start the log.
          </p>
        ) : (
          <ul className="text-xs">
            {[...events].reverse().map((entry) => (
              <li key={entry.id}>
                <button
                  type="button"
                  className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                  onClick={() => onRestore(entry)}
                >
                  {entry.id}. step {entry.step} {entry.label}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {events.length > 3 && (
        <div
          className="pointer-events-none absolute bottom-0 left-0 right-0 h-6 rounded-b-lg"
          style={{
            background: "linear-gradient(to bottom, transparent, var(--card))",
          }}
        />
      )}
    </div>
  );
}

// Fixed autoplay interval. Speed is not a dial: this sim advances in discrete
// training steps the reader drives manually; the interval only affects how fast
// autoplay runs, not the paper's ideas.
const TICK_INTERVAL_MS = 220;

// Advances the model by one training step every TICK_INTERVAL_MS, frame-rate
// independent, built on the shared animation-frame loop. Returns a reset for the
// accumulator so timing starts clean on play and reset.
function useTrainingClock(active: boolean, onTick: () => void): () => void {
  const accumulator = useRef(0);
  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (
      accumulator.current >= TICK_INTERVAL_MS &&
      steps < MAX_STEPS_PER_FRAME
    ) {
      accumulator.current -= TICK_INTERVAL_MS;
      onTick();
      steps += 1;
    }
  });
  return useCallback(() => {
    accumulator.current = 0;
  }, []);
}

// Draws the three learned behaviors over training so the reader watches tool use
// hold under MT-GRPO and slide under GRPO-OR.
function CurveChart({ state }: { state: SimState }) {
  const width = 1000;
  const height = 220;
  const padL = 36;
  const padR = 12;
  const padT = 14;
  const padB = 24;
  const points = state.history;
  const maxStep = Math.max(40, points[points.length - 1]?.step ?? 40);
  const x = (s: number) => padL + (s / maxStep) * (width - padL - padR);
  const y = (v: number) => padT + (1 - v) * (height - padT - padB);

  const series: {
    key: "toolUse" | "retrieval" | "answer";
    color: string;
    label: string;
  }[] = [
    { key: "toolUse", color: ACCENT, label: "tool use (turn 1)" },
    { key: "retrieval", color: SAGE, label: "retrieval (turn 1)" },
    { key: "answer", color: NEGATIVE, label: "answer (turn 2)" },
  ];

  function path(key: "toolUse" | "retrieval" | "answer"): string {
    return points
      .map(
        (point, i) =>
          `${i === 0 ? "M" : "L"} ${x(point.step).toFixed(1)} ${y(point[key]).toFixed(1)}`,
      )
      .join(" ");
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`Three behavior curves over ${maxStep} training steps under ${getAlgorithm(state.algorithm).name}. Tool use is ${percent(state.policy.toolUse)}, retrieval ${percent(state.policy.retrieval)}, answer accuracy ${percent(state.policy.answer)}.`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line
            x1={padL}
            y1={y(g)}
            x2={width - padR}
            y2={y(g)}
            className="stroke-foreground/10"
            strokeWidth={1}
          />
          <text
            x={padL - 6}
            y={y(g) + 3}
            textAnchor="end"
            className="fill-muted-foreground"
            style={{ fontSize: 10 }}
          >
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {series.map((s) => (
        <path
          key={s.key}
          d={path(s.key)}
          fill="none"
          stroke={s.color}
          strokeWidth={2.5}
          strokeLinejoin="round"
        />
      ))}
      {series.map((s, i) => (
        <g
          key={`legend-${s.key}`}
          transform={`translate(${padL + 8 + i * 230}, ${padT + 8})`}
        >
          <rect width={14} height={4} rx={2} fill={s.color} />
          <text
            x={20}
            y={5}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            {s.label}
          </text>
        </g>
      ))}
      <text
        x={width - padR}
        y={height - 6}
        textAnchor="end"
        className="fill-muted-foreground"
        style={{ fontSize: 11 }}
      >
        training step {state.step}
      </text>
    </svg>
  );
}

// The group of rollouts with the per-turn advantage each one earns under the
// chosen algorithm, the core thing the reader pokes.
function GroupTable({
  state,
  inspect,
  onInspect,
}: {
  state: SimState;
  inspect: number | null;
  onInspect: (id: number | null) => void;
}) {
  const advById = new Map(state.advantages.map((a) => [a.rolloutId, a]));
  function bar(value: number) {
    const magnitude = Math.min(1, Math.abs(value) / 2);
    return (
      <div className="flex h-3 items-center">
        <div className="flex h-full w-10 justify-end">
          {value < 0 && (
            <div
              className="h-full rounded-l-sm"
              style={{
                width: `${magnitude * 100}%`,
                backgroundColor: NEGATIVE,
              }}
            />
          )}
        </div>
        <div className="bg-foreground/30 h-full w-px" />
        <div className="flex h-full w-10">
          {value >= 0 && (
            <div
              className="h-full rounded-r-sm"
              style={{ width: `${magnitude * 100}%`, backgroundColor: ACCENT }}
            />
          )}
        </div>
      </div>
    );
  }

  const showFade = state.rollouts.length > 8;

  return (
    <div className="relative">
      <div className="ring-foreground/10 max-h-64 overflow-y-auto rounded-lg ring-1">
        <table className="w-full text-left text-xs">
          <thead className="bg-card text-muted-foreground sticky top-0">
            <tr>
              <th className="px-2 py-1 font-medium">#</th>
              <th className="px-2 py-1 font-medium">turn 1 (search)</th>
              <th className="px-2 py-1 font-medium">turn 2 (answer)</th>
              <th className="px-2 py-1 text-right font-medium">A₁</th>
              <th className="px-2 py-1 text-right font-medium">A₂</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {state.rollouts.map((rollout) => {
              const adv = advById.get(rollout.id);
              return (
                <tr
                  key={rollout.id}
                  className={
                    inspect === rollout.id
                      ? "bg-foreground/5"
                      : "text-muted-foreground"
                  }
                  onMouseEnter={() => onInspect(rollout.id)}
                  onFocus={() => onInspect(rollout.id)}
                  tabIndex={0}
                >
                  <td className="px-2 py-1">{rollout.id}</td>
                  <td className="px-2 py-1">
                    <span
                      className={rollout.toolExecuted ? "text-foreground" : ""}
                    >
                      {rollout.toolExecuted ? "tool ✓" : "no-tool"}
                    </span>
                    {" / "}
                    <span
                      className={rollout.retrieved ? "text-foreground" : ""}
                    >
                      {rollout.retrieved ? "found ✓" : "miss"}
                    </span>
                  </td>
                  <td className="px-2 py-1">
                    <span
                      className={rollout.answerCorrect ? "text-foreground" : ""}
                    >
                      {rollout.answerCorrect ? "correct ✓" : "wrong"}
                    </span>
                  </td>
                  <td className="px-2 py-1">{bar(adv?.turn1 ?? 0)}</td>
                  <td className="px-2 py-1">{bar(adv?.turn2 ?? 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {showFade && (
        <div
          className="pointer-events-none absolute bottom-0 left-0 right-0 h-8 rounded-b-lg"
          style={{
            background: "linear-gradient(to bottom, transparent, var(--card))",
          }}
        />
      )}
    </div>
  );
}

export function TurnLevelCredit() {
  const [presetId, setPresetId] = useState("mt-stable");
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialFromPreset("mt-stable"),
  );
  const [playing, setPlaying] = useState(false);
  const [inspect, setInspect] = useState<number | null>(null);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useTrainingClock(playing, () =>
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

  const setWeight = useCallback(
    (key: keyof RewardWeights, value: number) =>
      intervene({ type: "setWeight", key, value }),
    [intervene],
  );

  // Advance to the next seed value to show a different random sample. The
  // individual seed number has no pedagogical meaning; only the variety matters.
  const onResample = useCallback(() => {
    intervene({ type: "setSeed", value: (state.seed % 30) + 1 });
  }, [intervene, state.seed]);

  const algorithm = getAlgorithm(state.algorithm);
  const spread = turn1Spread(state.advantages);
  const hasSignal = turn1HasSignal(state.advantages);
  // What turn 1 would get under outcome-only credit, for the side-by-side.
  const orSpread = turn1Spread(
    computeAdvantages(state.rollouts, "grpo-or", state.weights),
  );

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
        {getPreset(presetId).blurb}
      </p>

      <div className="ring-foreground/10 overflow-hidden rounded-xl p-2 ring-1">
        <CurveChart state={state} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Credit assignment">
          <div className="flex flex-wrap gap-1.5">
            {ALGORITHMS.map((option) => (
              <Button
                key={option.id}
                type="button"
                size="sm"
                variant={state.algorithm === option.id ? "default" : "outline"}
                aria-pressed={state.algorithm === option.id}
                onClick={() =>
                  intervene({ type: "setAlgorithm", id: option.id })
                }
              >
                {option.name}
              </Button>
            ))}
          </div>
          <p className="text-muted-foreground text-xs">{algorithm.blurb}</p>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{algorithm.granularity}</Badge>
            <Badge
              variant="outline"
              style={{
                borderColor: hasSignal ? ACCENT : NEGATIVE,
                color: hasSignal ? ACCENT : NEGATIVE,
              }}
            >
              turn-1 signal {hasSignal ? "alive" : "dead"}
            </Badge>
            <Badge variant="outline">spread {spread.toFixed(2)}</Badge>
          </div>
          <p className="text-muted-foreground text-xs">
            Outcome-only credit would give turn 1 a spread of{" "}
            <span className="text-foreground font-mono">
              {orSpread.toFixed(2)}
            </span>{" "}
            on this group.
          </p>
          <GroupTable state={state} inspect={inspect} onInspect={setInspect} />
        </Panel>

        <Panel title="Turn rewards">
          <Slider
            label="Tool execution reward — points for turn 1 running the search tool"
            value={state.weights.toolExecution}
            min={0}
            max={0.5}
            step={0.05}
            display={state.weights.toolExecution.toFixed(2)}
            onChange={(value) => setWeight("toolExecution", value)}
          />
          <Slider
            label="Retrieval reward — bonus when the search returns the ground-truth answer"
            value={state.weights.retrieval}
            min={0}
            max={1}
            step={0.05}
            display={state.weights.retrieval.toFixed(2)}
            onChange={(value) => setWeight("retrieval", value)}
          />
          <Slider
            label="Search penalty λₛ — cost per search, discourages over-searching (drag to 0 to see collapse)"
            value={state.weights.searchPenalty}
            min={0}
            max={0.4}
            step={0.02}
            display={state.weights.searchPenalty.toFixed(2)}
            onChange={(value) => setWeight("searchPenalty", value)}
          />
          <Slider
            label="Exact-match reward — payoff when turn 2 produces the correct answer"
            value={state.weights.exactMatch}
            min={0.2}
            max={2}
            step={0.1}
            display={state.weights.exactMatch.toFixed(1)}
            onChange={(value) => setWeight("exactMatch", value)}
          />
          <Slider
            label="α — how much of the outcome advantage bleeds back into turn 1's credit (equation 5)"
            value={state.weights.alpha}
            min={0}
            max={1}
            step={0.05}
            display={state.weights.alpha.toFixed(2)}
            onChange={(value) => setWeight("alpha", value)}
          />
          <Slider
            label="Group size G — rollouts per update; larger groups stabilize normalization"
            value={state.groupSize}
            min={4}
            max={16}
            step={1}
            display={String(state.groupSize)}
            onChange={(value) => intervene({ type: "setGroupSize", value })}
          />
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-muted-foreground text-xs">
              Sample #{state.seed} — draw a different random group to check
              robustness
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onResample}
            >
              New sample
            </Button>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLog
              events={state.log}
              onRestore={(entry) =>
                intervene({
                  type: "restore",
                  snapshot: entry.snapshot,
                  label: `rewind to step ${entry.snapshot.step}`,
                })
              }
            />
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        The advantages here are exactly the paper&apos;s math: each turn&apos;s
        reward is group-normalized, A = (R &minus; mean) / std, and
        MT-GRPO&apos;s turn-1 advantage is A₁ = Aᴵ + α·Aᴼ (equation 5). Each
        preset starts from a fixed seed so the run is fully deterministic;
        &ldquo;New sample&rdquo; advances the seed to show a different random
        group. The training loop is a stand-in. Each step nudges the
        agent&apos;s behavior toward whatever the advantages reward, so
        unsignalled behavior drifts and forgets, which is why outcome-only
        credit lets tool use collapse. A real run trains a 7B model with a
        critic, not four propensities, but the credit-assignment story is the
        same one.
      </p>
    </div>
  );
}
