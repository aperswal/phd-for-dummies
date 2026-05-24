"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import {
  computeGeometry,
  initialState,
  methodName,
  PLANE_EXTENT,
  PRESETS,
  step,
  type GeometryEvent,
  type Intervention,
  type Method,
  type SimState,
} from "@/components/visualizations/on-sft-rl-and-on-policy-distillation-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7e5d";
const SLATE = "#7c8794";

const METHODS: { id: Method; label: string }[] = [
  { id: "rl", label: "RL" },
  { id: "sft", label: "SFT" },
  { id: "opd", label: "OPD" },
  { id: "opsd", label: "OPSD" },
];

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
  disabled?: boolean;
  onChange: (value: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step: stepSize,
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
        step={stepSize}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer disabled:opacity-40"
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
  events: GeometryEvent[];
  onRestore: (event: GeometryEvent) => void;
}) {
  return (
    <div className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1">
      {events.length === 0 ? (
        <p className="text-muted-foreground px-2 py-1.5 text-xs">
          Step, load a preset, or flip a control to start the log.
        </p>
      ) : (
        <ul className="text-xs">
          {[...events].reverse().map((event) => (
            <li key={event.id}>
              <button
                type="button"
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
  );
}

const VIEW = 320; // svg viewport size per panel
const CENTER = VIEW / 2;
const SCALE = (VIEW / 2 - 16) / PLANE_EXTENT;

function vectorColor(kind: "noise" | "biased" | "pivot"): string {
  if (kind === "pivot") return ACCENT;
  if (kind === "noise") return SLATE;
  return SAGE;
}

export function GradientGeometry() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("rl-cancels"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("rl-cancels");

  const { params, log } = state;
  const result = useMemo(() => computeGeometry(params), [params]);

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

  const safe = result.safeRadius * SCALE;
  const isOpsd = params.method === "opsd";
  const isRl = params.method === "rl";
  const isFan = params.method === "sft" || params.method === "opd";

  const updateTipX = CENTER + result.update.x * SCALE;
  const updateTipY = CENTER + result.update.y * SCALE;

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

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            viewBox={`0 0 ${VIEW} ${VIEW}`}
            className="w-full"
            role="img"
            aria-label={`The cloud of per-token gradient vectors for ${methodName(
              params.method,
            )} in the projected parameter plane.`}
          >
            <text
              x={12}
              y={20}
              className="fill-muted-foreground"
              style={{ fontSize: 11 }}
            >
              per-token gradients
            </text>
            <circle
              cx={CENTER}
              cy={CENTER}
              r={2.5}
              className="fill-foreground/40"
            />
            {result.gradients.map((g, index) => {
              const tipX = CENTER + g.vec.x * SCALE;
              const tipY = CENTER + g.vec.y * SCALE;
              return (
                <line
                  key={`g-${index}`}
                  x1={CENTER}
                  y1={CENTER}
                  x2={tipX}
                  y2={tipY}
                  stroke={vectorColor(g.kind)}
                  strokeWidth={g.kind === "pivot" ? 2.4 : 1}
                  strokeOpacity={g.kind === "pivot" ? 0.95 : 0.5}
                  strokeLinecap="round"
                />
              );
            })}
          </svg>
        </div>

        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            viewBox={`0 0 ${VIEW} ${VIEW}`}
            className="w-full"
            role="img"
            aria-label={`The batch-averaged update for ${methodName(
              params.method,
            )}. It ${
              result.collapsed
                ? "overshoots the safe radius, so the run collapses"
                : "stays inside the safe radius, so the step is usable"
            }.`}
          >
            <text
              x={12}
              y={20}
              className="fill-muted-foreground"
              style={{ fontSize: 11 }}
            >
              batch-averaged update
            </text>
            <circle
              cx={CENTER}
              cy={CENTER}
              r={safe}
              fill={
                result.collapsed
                  ? "rgba(194,104,63,0.06)"
                  : "rgba(111,126,93,0.06)"
              }
              stroke={result.collapsed ? ACCENT : SAGE}
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
            <text
              x={CENTER}
              y={CENTER - safe - 6}
              textAnchor="middle"
              fill={result.collapsed ? ACCENT : SAGE}
              style={{ fontSize: 11 }}
            >
              safe radius
            </text>
            <circle
              cx={CENTER}
              cy={CENTER}
              r={2.5}
              className="fill-foreground/40"
            />
            <line
              x1={CENTER}
              y1={CENTER}
              x2={updateTipX}
              y2={updateTipY}
              stroke={result.collapsed ? ACCENT : "var(--color-foreground)"}
              strokeWidth={3.2}
              strokeLinecap="round"
            />
            <circle
              cx={updateTipX}
              cy={updateTipY}
              r={4}
              fill={result.collapsed ? ACCENT : "var(--color-foreground)"}
            />
          </svg>
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
        <Panel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            {METHODS.map((method) => (
              <Toggle
                key={method.id}
                label={method.label}
                active={params.method === method.id}
                onClick={() =>
                  intervene({ type: "setMethod", method: method.id })
                }
              />
            ))}
          </div>
          <Toggle
            label="Per-token KL clip"
            active={params.clipping}
            onClick={() => intervene({ type: "toggleClipping" })}
          />
          <Slider
            label="Batch size (tokens averaged)"
            value={params.batchSize}
            min={4}
            max={200}
            step={2}
            display={String(Math.round(params.batchSize))}
            onChange={(value) => intervene({ type: "setBatchSize", value })}
          />
          <Slider
            label="Reward bias (RL signal vs noise)"
            value={params.rewardBias}
            min={0}
            max={3}
            step={0.1}
            display={params.rewardBias.toFixed(1)}
            disabled={!isRl}
            onChange={(value) => intervene({ type: "setRewardBias", value })}
          />
          <Slider
            label="Pivot strength (OPSD, x typical token)"
            value={params.pivotStrength}
            min={1}
            max={140}
            step={1}
            display={`${Math.round(params.pivotStrength)}x`}
            disabled={!isOpsd}
            onChange={(value) => intervene({ type: "setPivotStrength", value })}
          />
          <Slider
            label="Bias spread (SFT/OPD/OPSD fan)"
            value={params.spread}
            min={0.05}
            max={1.4}
            step={0.05}
            display={params.spread.toFixed(2)}
            disabled={isRl}
            onChange={(value) => intervene({ type: "setSpread", value })}
          />
          <Slider
            label="Seed (redraws the cloud)"
            value={params.seed}
            min={1}
            max={60}
            step={1}
            display={String(params.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </Panel>

        <Panel title="Inside the step">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{methodName(params.method)}</Badge>
            <Badge variant={result.collapsed ? "destructive" : "outline"}>
              {result.collapsed ? "collapse" : "stable"}
            </Badge>
            <Badge variant="outline">
              update {result.updateMagnitude.toFixed(1)} / {result.safeRadius}
            </Badge>
            <Badge variant="outline">
              concentration {result.concentration.toFixed(1)}x
            </Badge>
            <Badge variant="outline">
              spread {result.diffuseness.toFixed(2)}
            </Badge>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <dt className="text-muted-foreground">density</dt>
            <dd className="font-mono">{isRl ? "sparse" : "dense"}</dd>
            <dt className="text-muted-foreground">bias</dt>
            <dd className="font-mono">
              {isRl
                ? "unbiased"
                : isFan
                  ? params.method === "sft"
                    ? "toward data"
                    : "toward teacher"
                  : "toward self+hint"}
            </dd>
            <dt className="text-muted-foreground">concentration</dt>
            <dd className="font-mono">
              {isOpsd && !params.clipping ? "on pivot token" : "diffuse"}
            </dd>
            <dt className="text-muted-foreground">mean |grad|</dt>
            <dd className="font-mono">{result.meanMagnitude.toFixed(1)}</dd>
          </dl>
          <p className="text-muted-foreground text-xs">
            {isRl
              ? "Most vectors are noise. They cancel as the batch grows, leaving only the reward-aligned component."
              : isOpsd && !params.clipping
                ? "One pivot token (clay-orange) the teacher loves and the student barely supports drags the whole update off course."
                : isOpsd
                  ? "The pivot is capped, so the update is back to the diffuse pull the other dense methods share."
                  : "Every token pulls the same way, fanned out, so the update is large but spread."}
          </p>
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
        Each per-token gradient is projected from the model&apos;s
        high-dimensional parameter space onto a 2D plane so the cloud is
        legible; a real step lives in billions of dimensions. The vector
        distributions are hand-shaped to match the post&apos;s description of
        each method, and the &quot;safe radius&quot; is a stand-in for the
        largest step that keeps a run stable. The shapes are real; the numbers
        are illustrative.
      </p>
    </div>
  );
}
