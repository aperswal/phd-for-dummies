"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  initialState,
  isDone,
  MAX_ELO,
  MAX_ITERATIONS,
  MIN_ELO,
  MOHEX_ELO,
  PRESETS,
  readout,
  step,
  type ExitState,
  type Intervention,
  type TargetType,
} from "@/components/visualizations/thinking-fast-and-slow-with-deep-learning-and-tree-search-model";

const APPRENTICE = "#c2683f";
const EXPERT = "#6f8a5e";
const MOHEX = "#7c7368";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: ExitState, action: ViewAction): ExitState {
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
        style={{ accentColor: APPRENTICE }}
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

// A small fixed game tree drawn under the chart. The root has four candidate
// moves; the apprentice prior puts most of its mass on the second from the left.
// The bar over each move is the visit count the modified UCT formula would
// build, biased toward the prior when the prior is on. This is the per-position
// view of the same loop the chart shows iteration by iteration.
const PRIOR = [0.12, 0.55, 0.22, 0.11];
const RAW_VALUE = [0.3, 0.55, 0.62, 0.35];

function moveVisits(usePrior: boolean, priorWeight: number): number[] {
  const bias = usePrior ? Math.min(priorWeight / 100, 1.3) : 0;
  const scores = RAW_VALUE.map(
    (value, index) => value + bias * 0.6 * (PRIOR[index] ?? 0),
  );
  const total = scores.reduce((sum, value) => sum + value, 0) || 1;
  return scores.map((value) => value / total);
}

export function ExpertIterationLoop() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("exit"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("exit");

  const { params, history, log } = state;
  const stats = useMemo(() => readout(state), [state]);
  const done = useMemo(() => isDone(state), [state]);
  const visits = useMemo(
    () => moveVisits(params.useApprenticePrior, params.priorWeight),
    [params.useApprenticePrior, params.priorWeight],
  );
  const bestMove = visits.indexOf(Math.max(...visits));

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Keep a ref so the fixed-timestep callback (closed over on mount) can read
  // the latest done value without re-registering the interval on every tick.
  const doneRef = useRef(done);
  useEffect(() => {
    doneRef.current = done;
  }, [done]);

  const resetClock = useFixedTimestep(playing, 850 / speed, () => {
    if (doneRef.current) {
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

  // Chart layout. Iterations run along x, Elo up y, with the MoHeX line drawn as
  // a fixed reference the two curves either clear or stall under.
  const width = 1000;
  const height = 360;
  const padL = 52;
  const padR = 20;
  const padT = 20;
  const padB = 34;
  const maxIter = Math.max(40, history.length - 1);
  const xAt = (iter: number) =>
    padL + (iter / maxIter) * (width - padL - padR);
  const yAt = (elo: number) =>
    padT +
    (1 - (elo - MIN_ELO) / (MAX_ELO - MIN_ELO)) * (height - padT - padB);

  const linePath = (key: "apprentice" | "expert") =>
    history
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"} ${xAt(point.iteration).toFixed(1)} ${yAt(point[key]).toFixed(1)}`,
      )
      .join(" ");

  const isReinforce = params.mode === "reinforce";

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
          aria-label={`Strength over ${history.length - 1} iterations. The apprentice is at ${Math.round(state.apprentice)} Elo and the expert at ${Math.round(state.expert)} Elo, against a MoHeX reference of ${MOHEX_ELO} Elo.`}
        >
          {[-300, 0, 300, 600, 900, 1200, 1500].map((elo) => (
            <g key={`grid-${elo}`}>
              <line
                x1={padL}
                x2={width - padR}
                y1={yAt(elo)}
                y2={yAt(elo)}
                className="stroke-foreground/10"
                strokeWidth={1}
              />
              <text
                x={padL - 8}
                y={yAt(elo) + 4}
                textAnchor="end"
                className="fill-muted-foreground"
                style={{ fontSize: 11 }}
              >
                {elo}
              </text>
            </g>
          ))}

          <line
            x1={padL}
            x2={width - padR}
            y1={yAt(MOHEX_ELO)}
            y2={yAt(MOHEX_ELO)}
            stroke={MOHEX}
            strokeWidth={2}
            strokeDasharray="7 5"
          />
          <text
            x={width - padR}
            y={yAt(MOHEX_ELO) - 6}
            textAnchor="end"
            fill={MOHEX}
            style={{ fontSize: 11, fontWeight: 600 }}
          >
            MoHeX 1.0
          </text>

          {!isReinforce && (
            <path
              d={linePath("expert")}
              fill="none"
              stroke={EXPERT}
              strokeWidth={2.5}
            />
          )}
          <path
            d={linePath("apprentice")}
            fill="none"
            stroke={APPRENTICE}
            strokeWidth={2.5}
          />

          {!isReinforce && (
            <circle
              cx={xAt(state.iteration)}
              cy={yAt(state.expert)}
              r={4}
              fill={EXPERT}
            />
          )}
          <circle
            cx={xAt(state.iteration)}
            cy={yAt(state.apprentice)}
            r={4}
            fill={APPRENTICE}
          />

          <text
            x={padL}
            y={height - 8}
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            iteration {state.iteration}
          </text>
          <text
            x={width - padR}
            y={height - 8}
            textAnchor="end"
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            Elo strength
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={done}
        />
        <div className="min-w-[8rem] flex-1 sm:max-w-[12rem]">
          <Slider
            label="Playback pace"
            value={speed}
            min={0.5}
            max={3}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      {done && (
        <div className="ring-foreground/10 bg-muted/40 flex items-center justify-between rounded-lg px-4 py-2 ring-1">
          <span className="text-muted-foreground text-sm">
            Run complete — {MAX_ITERATIONS} iterations
          </span>
          <Button type="button" size="sm" variant="outline" onClick={onReset}>
            Reset
          </Button>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Tune the loop">
          <div className="flex flex-col gap-0.5">
            <span className="text-muted-foreground text-xs font-medium">
              Algorithm
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label="ExIt — apprentice imitates expert"
                active={params.mode === "exit"}
                onClick={() => intervene({ type: "setMode", value: "exit" })}
              />
              <Toggle
                label="REINFORCE — policy gradient only"
                active={params.mode === "reinforce"}
                onClick={() =>
                  intervene({ type: "setMode", value: "reinforce" })
                }
              />
            </div>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-muted-foreground text-xs font-medium">
              Imitation target
            </span>
            <div className="flex flex-wrap gap-1.5">
              {(["TPT", "CAT"] as TargetType[]).map((target) => (
                <Toggle
                  key={target}
                  label={`${target}${target === "TPT" ? " — match visit distribution" : " — match best move only"}`}
                  active={params.target === target}
                  onClick={() =>
                    intervene({ type: "setTarget", value: target })
                  }
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-muted-foreground text-xs font-medium">
              Search components
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label="Apprentice prior — learned hunch steers search"
                active={params.useApprenticePrior}
                onClick={() => intervene({ type: "toggleApprenticePrior" })}
              />
              <Toggle
                label="Value net — shortcut rollouts with learned value"
                active={params.useValueNet}
                onClick={() => intervene({ type: "toggleValueNet" })}
              />
            </div>
          </div>
          <Slider
            label="MCTS simulations per move — more gives a stronger expert"
            value={params.simulations}
            min={500}
            max={30000}
            step={500}
            display={String(params.simulations)}
            onChange={(value) => intervene({ type: "setSimulations", value })}
          />
          <Slider
            label="Prior weight w_a — how strongly the hunch steers search (paper: 100)"
            value={params.priorWeight}
            min={0}
            max={300}
            step={10}
            display={String(params.priorWeight)}
            onChange={(value) => intervene({ type: "setPriorWeight", value })}
          />
          <Slider
            label="Dataset size per iteration — more positions close more of the gap"
            value={params.datasetSize}
            min={24000}
            max={550000}
            step={1000}
            display={`${Math.round(params.datasetSize / 1000)}k`}
            onChange={(value) => intervene({ type: "setDatasetSize", value })}
          />
        </SimPanel>

        <SimPanel title="Inside one search">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="outline">
              apprentice {Math.round(state.apprentice)} Elo
            </Badge>
            <Badge variant="outline">
              expert {Math.round(state.expert)} Elo
            </Badge>
            <Badge variant="outline">gap {Math.round(stats.gap)} Elo</Badge>
            <Badge variant={stats.beatsMohex ? "default" : "outline"}>
              {stats.beatsMohex
                ? `✓ beats MoHeX (iter ${stats.iterationsToMohex})`
                : "below MoHeX"}
            </Badge>
          </div>

          <svg
            viewBox="0 0 300 150"
            className="w-full"
            role="img"
            aria-label={`The expert reads four candidate moves. The search spends most of its visits on move ${bestMove + 1}.`}
          >
            <line
              x1={150}
              y1={18}
              x2={45}
              y2={70}
              className="stroke-foreground/30"
            />
            <line
              x1={150}
              y1={18}
              x2={115}
              y2={70}
              className="stroke-foreground/30"
            />
            <line
              x1={150}
              y1={18}
              x2={185}
              y2={70}
              className="stroke-foreground/30"
            />
            <line
              x1={150}
              y1={18}
              x2={255}
              y2={70}
              className="stroke-foreground/30"
            />
            <circle cx={150} cy={14} r={7} fill={MOHEX} />
            {visits.map((visit, index) => {
              const cx = 45 + index * 70;
              const barH = visit * 60;
              const isBest = index === bestMove;
              return (
                <g key={`move-${index}`}>
                  <circle
                    cx={cx}
                    cy={74}
                    r={6}
                    fill={isBest ? APPRENTICE : "var(--color-muted)"}
                    stroke={isBest ? APPRENTICE : "transparent"}
                    strokeWidth={2}
                  />
                  <rect
                    x={cx - 9}
                    y={140 - barH}
                    width={18}
                    height={barH}
                    rx={2}
                    fill={isBest ? APPRENTICE : EXPERT}
                    fillOpacity={isBest ? 1 : 0.55}
                  />
                  <text
                    x={cx}
                    y={140 - barH - 4}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 9 }}
                  >
                    {Math.round(visit * 100)}%
                  </text>
                </g>
              );
            })}
            <text
              x={5}
              y={148}
              className="fill-muted-foreground"
              style={{ fontSize: 9 }}
            >
              visits per move
            </text>
          </svg>
          <p className="text-muted-foreground text-xs">
            {params.useApprenticePrior
              ? "The apprentice prior pulls the search toward move 2, so the visits concentrate instead of spreading evenly."
              : "With the prior off, the search splits its visits by raw reading alone and explores more widely."}
          </p>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Play, step an iteration, or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to iter ${event.snapshot.iteration}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        Strength here is an Elo recurrence that follows the paper&apos;s
        framing. The expert is the apprentice plus a search gain, and each
        iteration the apprentice closes part of the gap up to the expert it just
        imitated. The run is deterministic (fixed seed) and stops at{" "}
        {MAX_ITERATIONS} iterations. The real ExIt ran 9x9 Hex with 10,000
        simulations per move over hundreds of thousands of positions; this
        compresses the loop so the compounding is watchable, not at the
        paper&apos;s scale.
      </p>
    </div>
  );
}
