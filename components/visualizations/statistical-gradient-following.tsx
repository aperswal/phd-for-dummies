"use client";

import { useCallback, useMemo, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  globalPeak,
  gradientMu,
  initialState,
  PRESETS,
  rawReward,
  status,
  step,
  type BaselineMode,
  type Intervention,
  type SigmaMode,
  type SimState,
  type Vec2,
} from "@/components/visualizations/statistical-gradient-following-model";

const ACCENT = "#c2683f";
const SLATE = "#7c8a99";
const SAGE = "#6f7d5f";
const VIEW = 100;
const HEAT_RES = 30;

// Fixed seed so Reset always replays the same trajectory as first load. The
// individual seed value carries no meaning the reader could observe; only the
// run's determinism matters.
const INITIAL_SEED = 1;

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string; seed: number };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId, action.seed);
  return step(state, action);
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  disabled,
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
        disabled={disabled}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
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

// A short polyline of recent batch rewards, so the reader watches the climb (or
// the jitter) as a curve. Scaled to its own min and max so any landscape fits.
function RewardSparkline({ history }: { history: number[] }) {
  if (history.length < 2) {
    return (
      <p className="text-muted-foreground text-xs">
        Run a few batches to draw the reward curve.
      </p>
    );
  }
  const lo = Math.min(...history);
  const hi = Math.max(...history);
  const span = hi - lo < 1e-9 ? 1 : hi - lo;
  const points = history
    .map((value, index) => {
      const x = (index / (history.length - 1)) * 100;
      const y = 24 - ((value - lo) / span) * 22 - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox="0 0 100 24"
      className="h-10 w-full"
      role="img"
      aria-label="Recent reward per batch"
    >
      <polyline points={points} fill="none" stroke={ACCENT} strokeWidth={1.4} />
    </svg>
  );
}

function arrowTip(from: Vec2, dir: Vec2, length: number): Vec2 {
  const mag = Math.hypot(dir.x, dir.y);
  if (mag < 1e-9) return from;
  return {
    x: from.x + (dir.x / mag) * length,
    y: from.y + (dir.y / mag) * length,
  };
}

export function GaussianSearch() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("climb", INITIAL_SEED),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [presetId, setPresetId] = useState("climb");
  const [inspect, setInspect] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { landscape, params, mu, sigma, samples, log } = state;
  const score = useMemo(() => status(state), [state]);
  const peak = useMemo(() => globalPeak(landscape), [landscape]);

  // Once the unit has settled on the global peak it has learned everything it
  // can. Guard every advancement path so the sim stops without a state update
  // inside an effect, which the linter disallows.
  const isDone = score.atGlobalPeak;

  // The heatmap is the landscape the reader can see but the unit cannot. It only
  // changes when the landscape does, so it is memoised away from the running sim.
  const heat = useMemo(() => {
    const cells: { x: number; y: number; t: number }[] = [];
    let lo = Infinity;
    let hi = -Infinity;
    const raw: number[] = [];
    for (let row = 0; row < HEAT_RES; row++) {
      for (let col = 0; col < HEAT_RES; col++) {
        const p = { x: (col + 0.5) / HEAT_RES, y: (row + 0.5) / HEAT_RES };
        const value = rawReward(landscape, p);
        raw.push(value);
        lo = Math.min(lo, value);
        hi = Math.max(hi, value);
      }
    }
    const span = hi - lo < 1e-9 ? 1 : hi - lo;
    let index = 0;
    for (let row = 0; row < HEAT_RES; row++) {
      for (let col = 0; col < HEAT_RES; col++) {
        cells.push({
          x: (col / HEAT_RES) * VIEW,
          y: (row / HEAT_RES) * VIEW,
          t: ((raw[index] ?? lo) - lo) / span,
        });
        index += 1;
      }
    }
    return cells;
  }, [landscape]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !isDone, 360 / speed, () =>
    dispatch({ kind: "tick" }),
  );

  const onPlayPause = useCallback(() => {
    if (isDone) return;
    resetClock();
    setPlaying((value) => !value);
  }, [isDone, resetClock]);
  const onStep = useCallback(() => {
    if (isDone) return;
    dispatch({ kind: "tick" });
  }, [isDone]);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    setInspect(null);
    dispatch({ kind: "reset", presetId, seed: INITIAL_SEED });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      setInspect(null);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  // Click anywhere on the landscape to drop the aim there. The svg uses a 0..100
  // viewBox, so a click is mapped from its rendered box back into that space.
  const onCanvasClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      intervene({ type: "moveMu", pos: { x, y } });
    },
    [intervene],
  );

  const cx = mu.x * VIEW;
  const cy = mu.y * VIEW;
  const sigmaR = sigma * VIEW;

  const gradDir = gradientMu(landscape, mu, sigma);
  const gradTip = arrowTip(
    { x: cx, y: cy },
    { x: gradDir.x, y: gradDir.y },
    17,
  );
  const stepTip = arrowTip(
    { x: cx, y: cy },
    { x: state.avgUpdateMu.x, y: state.avgUpdateMu.y },
    14,
  );
  const stepMag = Math.hypot(state.avgUpdateMu.x, state.avgUpdateMu.y);

  const inspected =
    inspect !== null && inspect < samples.length ? samples[inspect] : null;

  const aligned = score.gradientCos > 0.25;
  const verdict = score.atGlobalPeak
    ? { label: "on the best peak", color: SAGE }
    : score.trappedOnLesserPeak
      ? { label: "trapped on a lesser peak", color: ACCENT }
      : { label: "searching", color: undefined };

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

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            ref={svgRef}
            viewBox={`-4 -4 ${VIEW + 8} ${VIEW + 8}`}
            className="w-full cursor-crosshair touch-none"
            role="img"
            aria-label={`A reward landscape the unit cannot see. Its aim is at ${mu.x.toFixed(2)}, ${mu.y.toFixed(2)} with exploration width ${sigma.toFixed(2)}. Expected reward is ${pct(score.fractionOfBest)} of the best peak.`}
            onClick={onCanvasClick}
          >
            {heat.map((cell, index) => (
              <rect
                key={index}
                x={cell.x}
                y={cell.y}
                width={VIEW / HEAT_RES + 0.4}
                height={VIEW / HEAT_RES + 0.4}
                fill={ACCENT}
                fillOpacity={0.05 + 0.62 * cell.t}
              />
            ))}

            {/* The ring marks the tallest hill so the reader can see the goal;
                the unit never has access to this location. Using the background
                color so it stays visible against the heatmap in both themes. */}
            <circle
              cx={peak.center.x * VIEW}
              cy={peak.center.y * VIEW}
              r={2.4}
              fill="none"
              stroke="var(--color-background)"
              strokeWidth={1.5}
              strokeOpacity={0.9}
            />

            {/* The exploration spread: every try is drawn from inside this circle. */}
            <circle
              cx={cx}
              cy={cy}
              r={sigmaR}
              fill="none"
              stroke="var(--color-foreground)"
              strokeOpacity={0.45}
              strokeWidth={0.8}
              strokeDasharray="2 2"
            />

            {samples.map((sample, index) => {
              const good = sample.advantage > 0;
              return (
                <circle
                  key={index}
                  cx={sample.pos.x * VIEW}
                  cy={sample.pos.y * VIEW}
                  r={index === inspect ? 2.4 : 1.5}
                  fill={good ? ACCENT : "none"}
                  stroke={good ? ACCENT : SLATE}
                  strokeWidth={1}
                  onMouseEnter={() => setInspect(index)}
                  onFocus={() => setInspect(index)}
                  className="cursor-pointer"
                />
              );
            })}

            {/* True uphill direction (faint ink) and the averaged step (clay
                orange). When they point the same way, the unit climbs E{r}. */}
            <line
              x1={cx}
              y1={cy}
              x2={gradTip.x}
              y2={gradTip.y}
              stroke="var(--color-foreground)"
              strokeOpacity={0.5}
              strokeWidth={1}
              strokeDasharray="2 1.5"
              markerEnd="url(#sgf-grad)"
            />
            {stepMag > 1e-7 && (
              <line
                x1={cx}
                y1={cy}
                x2={stepTip.x}
                y2={stepTip.y}
                stroke={ACCENT}
                strokeWidth={1.8}
                strokeLinecap="round"
                markerEnd="url(#sgf-step)"
              />
            )}

            <circle cx={cx} cy={cy} r={2.4} fill="var(--color-foreground)" />
            <circle
              cx={cx}
              cy={cy}
              r={4}
              fill="none"
              stroke="var(--color-foreground)"
              strokeWidth={1.2}
            />

            <defs>
              <marker
                id="sgf-grad"
                viewBox="0 0 10 10"
                refX="7"
                refY="5"
                markerWidth="4"
                markerHeight="4"
                orient="auto-start-reverse"
              >
                <path
                  d="M 0 1 L 8 5 L 0 9"
                  fill="none"
                  stroke="var(--color-foreground)"
                  strokeOpacity={0.5}
                  strokeWidth={2}
                />
              </marker>
              <marker
                id="sgf-step"
                viewBox="0 0 10 10"
                refX="7"
                refY="5"
                markerWidth="4.5"
                markerHeight="4.5"
                orient="auto-start-reverse"
              >
                <path
                  d="M 0 1 L 8 5 L 0 9"
                  fill="none"
                  stroke={ACCENT}
                  strokeWidth={2}
                />
              </marker>
            </defs>
          </svg>
        </div>

        <SimPanel title="What the unit has learned" className="lg:self-start">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">batch {state.batches}</Badge>
            <Badge variant="outline">{state.trials} tries</Badge>
            <Badge variant="outline">reward {pct(score.fractionOfBest)}</Badge>
            <Badge variant="outline">spread {sigma.toFixed(3)}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge
              variant={aligned ? "default" : "outline"}
              style={
                aligned ? { backgroundColor: SAGE, color: "white" } : undefined
              }
            >
              step {aligned ? "follows" : "off"} gradient (
              {score.gradientCos.toFixed(2)})
            </Badge>
            <Badge
              variant={verdict.color ? "default" : "outline"}
              style={
                verdict.color
                  ? { backgroundColor: verdict.color, color: "white" }
                  : undefined
              }
            >
              {verdict.label}
            </Badge>
          </div>

          <RewardSparkline history={state.rewardHistory} />

          {inspected ? (
            <p className="text-muted-foreground text-xs">
              this try landed at ({inspected.pos.x.toFixed(2)},{" "}
              {inspected.pos.y.toFixed(2)}), scored{" "}
              <span className="text-foreground font-mono">
                {inspected.reward.toFixed(2)}
              </span>
              , so it{" "}
              <span className="text-foreground font-mono">
                {inspected.advantage > 0 ? "pulls" : "pushes"}
              </span>{" "}
              the aim {inspected.advantage > 0 ? "toward" : "away from"} it
              (advantage {inspected.advantage.toFixed(2)}).
            </p>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click the landscape to drop the aim anywhere. Hover a try to read
              its score. Filled orange tries beat the baseline and pull the aim;
              hollow slate tries fall short and push it away.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Reach a peak or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to batch ${event.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing && !isDone}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        {isDone && (
          <span
            className="text-xs font-medium"
            style={{ color: SAGE }}
            aria-live="polite"
          >
            Converged — on the best peak. Reset to run again.
          </span>
        )}
        <div className="min-w-[8rem] flex-1 sm:max-w-[10rem]">
          <Slider
            label="Playback speed (animation pace, not the algorithm)"
            value={speed}
            min={1}
            max={12}
            step={1}
            display={`${speed}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <SimPanel title="Paper levers">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">baseline</span>
          {(["comparison", "none"] as BaselineMode[]).map((mode) => (
            <Toggle
              key={mode}
              label={
                mode === "comparison"
                  ? "reinforcement comparison"
                  : "none (b = 0)"
              }
              active={params.baselineMode === mode}
              onClick={() =>
                intervene({ type: "setBaselineMode", value: mode })
              }
            />
          ))}
          <span className="text-muted-foreground ml-2 text-xs">spread</span>
          {(["adapt", "fixed"] as SigmaMode[]).map((mode) => (
            <Toggle
              key={mode}
              label={mode === "adapt" ? "adapt sigma" : "pin sigma"}
              active={params.sigmaMode === mode}
              onClick={() => intervene({ type: "setSigmaMode", value: mode })}
            />
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="alpha — how far the aim moves each batch"
            value={params.alpha}
            min={0.02}
            max={1}
            step={0.02}
            display={params.alpha.toFixed(2)}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          <Slider
            label="sigma — how widely the unit throws its tries"
            value={sigma}
            min={0.012}
            max={0.45}
            step={0.005}
            display={sigma.toFixed(3)}
            onChange={(value) => intervene({ type: "setSigma", value })}
          />
          <Slider
            label="baseline decay — how quickly the average tracks recent reward"
            value={params.baselineDecay}
            min={0.02}
            max={0.6}
            step={0.02}
            display={params.baselineDecay.toFixed(2)}
            disabled={params.baselineMode === "none"}
            onChange={(value) => intervene({ type: "setBaselineDecay", value })}
          />
          <Slider
            label="reward noise — corrupts each score so the baseline must work harder"
            value={params.rewardNoise}
            min={0}
            max={0.6}
            step={0.02}
            display={params.rewardNoise.toFixed(2)}
            onChange={(value) => intervene({ type: "setRewardNoise", value })}
          />
        </div>
        <Button
          type="button"
          size="sm"
          variant="secondary"
          onClick={() => intervene({ type: "resetLearning" })}
        >
          Reset learning (keep the hill)
        </Button>
      </SimPanel>

      <p className="text-muted-foreground text-xs">
        Each batch draws {params.batchSize} tries from N(aim, sigma squared),
        scores them on the hidden landscape, and applies the paper&apos;s
        averaged update, &Delta;&mu; = &alpha;(r &minus; b)(y &minus; &mu;) and
        the matching rule for sigma. The orange arrow is the unit&apos;s averaged
        step; the faint dashed arrow is the true gradient of expected reward,
        computed separately only to check that the averaged step aligns with it
        (Theorem 1). The ring on the heatmap marks the tallest hill, which the
        unit never sees. The run uses a fixed seed; Reset replays the same
        trajectory.
      </p>
    </div>
  );
}
