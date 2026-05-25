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
  AXES,
  buildCurve,
  computeAllocation,
  getAxis,
  initialState,
  POINT_COUNT,
  PRESETS,
  step,
  type AxisId,
  type Intervention,
  type SimEvent,
  type SimState,
} from "@/components/visualizations/scaling-laws-model";

const ACCENT = "#c2683f";
const SAGE = "#7f9472";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

// Fixed-timestep transport inlined so this simulation is one self-contained
// unit. It accumulates elapsed ms and fires onTick once per whole interval,
// capping catch-up so a backgrounded tab can't burst.
const MAX_STEPS_PER_FRAME = 240;

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

function EventLog({
  events,
  onRestore,
}: {
  events: SimEvent[];
  onRestore: (event: SimEvent) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Autoscroll to the top whenever new events arrive (list is newest-first).
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length]);

  return (
    <div className="relative">
      <div
        ref={scrollRef}
        className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1"
      >
        {events.length === 0 ? (
          <p className="text-muted-foreground px-2 py-1.5 text-xs">
            Step, flip the axis, or drag a control to start the log.
          </p>
        ) : (
          <ul className="text-xs">
            {[...events].reverse().map((event) => (
              <li key={event.id}>
                <button
                  type="button"
                  aria-label={`Restore to step ${event.id}: ${event.label}`}
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
      {/* Fade affordance signals hidden content below when the log overflows */}
      {events.length > 3 && (
        <div
          className="from-background pointer-events-none absolute right-0 bottom-0 left-0 h-6 rounded-b-lg bg-gradient-to-t to-transparent"
          aria-hidden
        />
      )}
    </div>
  );
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

function formatScale(value: number): string {
  if (value === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / Math.pow(10, exp);
  return `${mantissa.toFixed(1)}e${exp}`;
}

// Map a (logX, logLoss) data point into SVG pixel space on a log-log plot.
interface PlotBox {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  logXMin: number;
  logXMax: number;
  logLossMin: number;
  logLossMax: number;
}

function project(box: PlotBox, logX: number, logLoss: number) {
  const tx = (logX - box.logXMin) / (box.logXMax - box.logXMin);
  const ty = (logLoss - box.logLossMin) / (box.logLossMax - box.logLossMin);
  return {
    px: box.x0 + tx * (box.x1 - box.x0),
    py: box.y1 - ty * (box.y1 - box.y0),
  };
}

export function ScalingLawsExplorer() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("params-law"),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState("params-law");

  const { params, revealLog, log } = state;
  const axis = getAxis(params.axisId);
  const isCompute = params.axisId === "compute";

  const curve = useMemo(
    () => buildCurve(params.axisId, revealLog, params.seed, POINT_COUNT),
    [params.axisId, params.seed, revealLog],
  );
  const fullCurve = useMemo(
    () => buildCurve(params.axisId, axis.max, params.seed, POINT_COUNT),
    [params.axisId, params.seed, axis.max],
  );
  const allocation = useMemo(() => computeAllocation(params), [params]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Stop the playback at the moment the curve finishes drawing so it doesn't
  // spin idle. The "at end" flag is mirrored into a ref inside an effect (the
  // same pattern useAnimationFrame uses), so the tick callback reads it without
  // any setState or ref write during render.
  const atEnd = revealLog >= axis.max - 1e-9;
  const atEndRef = useRef(atEnd);
  useEffect(() => {
    atEndRef.current = atEnd;
  }, [atEnd]);

  const resetClock = useFixedTimestep(playing, 520 / 1.5, () => {
    if (atEndRef.current) {
      setPlaying(false);
      return;
    }
    dispatch({ kind: "tick" });
  });

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

  const setAxis = useCallback(
    (id: AxisId) => {
      setPlaying(false);
      resetClock();
      intervene({ type: "setAxis", id });
    },
    [intervene, resetClock],
  );

  // Plot geometry. A log-log box with the loss on the vertical axis and the
  // chosen scale factor on the horizontal, both in powers of ten.
  const width = 720;
  const height = 360;
  const logLossValues = fullCurve.map((point) => point.logLoss);
  const logLossMin = Math.min(...logLossValues) - 0.05;
  const logLossMax = Math.max(...logLossValues) + 0.05;
  const box: PlotBox = {
    x0: 64,
    x1: width - 24,
    y0: 24,
    y1: height - 44,
    logXMin: axis.min,
    logXMax: axis.max,
    logLossMin,
    logLossMax,
  };

  const linePath = curve
    .map((point, i) => {
      const { px, py } = project(box, point.logX, point.logLoss);
      return `${i === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`;
    })
    .join(" ");

  const xTicks = [];
  for (let e = Math.ceil(axis.min); e <= Math.floor(axis.max); e++) {
    xTicks.push(e);
  }
  const yTicks = [];
  for (
    let e = Math.ceil(logLossMin * 10) / 10;
    e <= logLossMax;
    e += (logLossMax - logLossMin) / 4
  ) {
    yTicks.push(e);
  }

  const excessPct = Math.round(allocation.excess * 100);
  const overfitPct = Math.round(allocation.overfit * 100);
  const splitHealthy = allocation.excess < 0.08;

  const onReshuffle = useCallback(() => {
    // Pick a new seed so the empirical jitter looks different. The seed itself
    // is arbitrary — only the visual scatter changes, not the power-law trend.
    const newSeed = Math.floor(Math.random() * 1000) + 1;
    intervene({ type: "reshuffle", seed: newSeed });
  }, [intervene]);

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      {/* Spine: scenario presets are the primary controls */}
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
          aria-label={`Log-log plot of test loss versus ${axis.label}. The loss falls along a straight power law with slope ${axis.alpha}.`}
        >
          <rect
            x={box.x0}
            y={box.y0}
            width={box.x1 - box.x0}
            height={box.y1 - box.y0}
            className="fill-foreground/[0.02]"
          />
          {xTicks.map((e) => {
            const { px } = project(box, e, logLossMin);
            return (
              <g key={`xt-${e}`}>
                <line
                  x1={px}
                  y1={box.y0}
                  x2={px}
                  y2={box.y1}
                  className="stroke-foreground/5"
                />
                <text
                  x={px}
                  y={box.y1 + 16}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 10 }}
                >
                  1e{e}
                </text>
              </g>
            );
          })}
          {yTicks.map((e, i) => {
            const { py } = project(box, axis.min, e);
            return (
              <g key={`yt-${i}`}>
                <line
                  x1={box.x0}
                  y1={py}
                  x2={box.x1}
                  y2={py}
                  className="stroke-foreground/5"
                />
                <text
                  x={box.x0 - 6}
                  y={py + 3}
                  textAnchor="end"
                  className="fill-muted-foreground"
                  style={{ fontSize: 10 }}
                >
                  {Math.pow(10, e).toFixed(1)}
                </text>
              </g>
            );
          })}

          <path
            d={linePath}
            fill="none"
            stroke={ACCENT}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {curve.map((point, i) => {
            const { px, py } = project(box, point.logX, point.logLoss);
            return (
              <circle
                key={`pt-${i}`}
                cx={px}
                cy={py}
                r={2.6}
                fill={ACCENT}
                stroke="var(--color-card)"
                strokeWidth={1}
              />
            );
          })}

          <text
            x={box.x0}
            y={box.y0 - 8}
            className="fill-foreground"
            style={{ fontSize: 12, fontWeight: 600 }}
          >
            test loss
          </text>
          <text
            x={(box.x0 + box.x1) / 2}
            y={height - 6}
            textAnchor="middle"
            className="fill-foreground"
            style={{ fontSize: 12, fontWeight: 600 }}
          >
            {axis.label} ({axis.unit})
          </text>
        </svg>
      </div>

      {/* Transport row: playback controls + axis switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="flex flex-wrap gap-1.5">
          {AXES.map((option) => (
            <Button
              key={option.id}
              type="button"
              size="sm"
              variant={params.axisId === option.id ? "default" : "outline"}
              aria-pressed={params.axisId === option.id}
              onClick={() => setAxis(option.id)}
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Compute budget allocator">
          <p className="text-muted-foreground text-xs">{axis.blurb}</p>
          <Slider
            label="Compute budget (log₁₀ PF-days)"
            value={params.logCompute}
            min={-4}
            max={4}
            step={0.5}
            display={`1e${params.logCompute.toFixed(1)} PF-days`}
            onChange={(value) => intervene({ type: "setCompute", value })}
          />
          <Slider
            label="Allocation — how far to deviate from the optimal model size N* ~ C^0.73 (1 = optimal split)"
            value={params.allocation}
            min={0.05}
            max={1.8}
            step={0.01}
            display={`${params.allocation.toFixed(2)}x`}
            onChange={(value) => intervene({ type: "setAllocation", value })}
          />
          <div className="flex items-center gap-2 text-xs">
            {/* Color + glyph so the meaning is never color-alone */}
            <span
              className="inline-flex size-2.5 rounded-full"
              style={{ backgroundColor: splitHealthy ? SAGE : ACCENT }}
              aria-hidden
            />
            <span className="text-muted-foreground">
              {splitHealthy
                ? "✓ near the optimal split"
                : `✗ ${excessPct}% worse loss than the best split`}
            </span>
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={onReshuffle}
          >
            Reshuffle empirical scatter
          </Button>
        </Panel>

        <Panel title="Inside the allocation">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              loss {allocation.realizedLoss.toFixed(3)}
            </Badge>
            <Badge variant="outline">
              best {allocation.optimalLoss.toFixed(3)}
            </Badge>
            {/* variant alone is color-only; add a glyph so meaning is explicit */}
            <Badge variant={overfitPct > 5 ? "default" : "outline"}>
              {overfitPct > 5 ? "⚠ " : ""}overfit +{overfitPct}%
            </Badge>
          </div>
          <div className="ring-foreground/10 rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <tbody className="font-mono">
                <tr className="border-foreground/5 border-b">
                  <td className="text-muted-foreground px-2 py-1.5">
                    compute C
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    {formatScale(allocation.compute)} PF-days
                  </td>
                </tr>
                <tr className="border-foreground/5 border-b">
                  <td className="text-muted-foreground px-2 py-1.5">
                    optimal N*
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    {formatScale(allocation.optimalN)}
                  </td>
                </tr>
                <tr className="border-foreground/5 border-b">
                  <td className="text-muted-foreground px-2 py-1.5">
                    your model N
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    {/* Color + glyph so off-optimal is never color-alone */}
                    {splitHealthy ? null : (
                      <span className="mr-1" style={{ color: ACCENT }}>
                        ✗
                      </span>
                    )}
                    {formatScale(allocation.chosenN)}
                  </td>
                </tr>
                <tr>
                  <td className="text-muted-foreground px-2 py-1.5">
                    tokens D
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    {formatScale(allocation.tokens)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLog
              events={log.events}
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
        The curves come straight from the paper&apos;s fitted power laws,{" "}
        {isCompute
          ? "L(C_min) = (Cc / C)^0.050"
          : params.axisId === "data"
            ? "L(D) = (Dc / D)^0.095"
            : "L(N) = (Nc / N)^0.076"}
        , and the allocator uses the combined L(N, D) fit with the optimal split
        N* &asymp; C^0.73. The numbers are the paper&apos;s WebText2 fits, not a
        fresh training run. The empirical scatter on the points is seeded and
        deterministic per preset; press &ldquo;Reshuffle empirical scatter&rdquo;
        to see a different noise sample, which illustrates that the power-law
        trend holds regardless of which noise draw you pick.
      </p>
    </div>
  );
}
