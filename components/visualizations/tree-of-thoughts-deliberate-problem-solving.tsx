"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  childrenOf,
  getPreset,
  initialState,
  PRESETS,
  PUZZLES,
  step,
  type Intervention,
  type NodeStatus,
  type SimState,
  type ThoughtNode,
  type Verdict,
} from "@/components/visualizations/tree-of-thoughts-deliberate-problem-solving-model";

const ACCENT = "#c2683f";
const SAGE = "#7c8a6b";
const SLATE = "#64748b";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") {
    const preset = getPreset(action.presetId);
    return initialState(preset.params, preset.seed);
  }
  if (action.kind === "tick") return step(state, { kind: "tick" });
  return step(state, { kind: "intervene", action: action.action });
}

const STATUS_FILL: Record<NodeStatus, string> = {
  frontier: "var(--color-muted)",
  expanded: "var(--color-muted)",
  kept: ACCENT,
  pruned: SLATE,
  solution: "#2f7d4f",
  dead: SLATE,
};

const VERDICT_LABEL: Record<Verdict, string> = {
  sure: "sure",
  maybe: "maybe",
  impossible: "no",
};

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  disabled = false,
}: SliderProps) {
  return (
    <label
      className={`flex flex-col gap-1 text-xs${disabled ? " opacity-50" : ""}`}
    >
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
        onChange={(event) => onChange(Number(event.target.value))}
        className={`h-1.5 w-full${disabled ? " cursor-not-allowed" : " cursor-pointer"}`}
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

interface Placed {
  node: ThoughtNode;
  x: number;
  y: number;
}

// Lay the tree out depth by depth, spacing each level's nodes evenly across the
// width so the branching reads as a tree. Pure layout, no model state.
function layout(state: SimState, width: number, height: number): Placed[] {
  const byDepth = new Map<number, ThoughtNode[]>();
  for (const node of state.nodes) {
    const list = byDepth.get(node.depth) ?? [];
    list.push(node);
    byDepth.set(node.depth, list);
  }
  const maxDepth = Math.max(...state.nodes.map((n) => n.depth));
  const placed: Placed[] = [];
  const topPad = 34;
  const rowGap = maxDepth > 0 ? (height - topPad - 24) / maxDepth : 0;
  for (const [depth, nodes] of byDepth) {
    const ordered = nodes.slice().sort((a, b) => a.id - b.id);
    const slot = width / (ordered.length + 1);
    ordered.forEach((node, index) => {
      placed.push({
        node,
        x: slot * (index + 1),
        y: topPad + depth * rowGap,
      });
    });
  }
  return placed;
}

// Pick the small label shown above each non-root node. Misjudged takes
// priority over pruned so the paper's key failure mode is always named.
function nodeLabel(node: ThoughtNode): string {
  if (node.status === "solution") return "solved ✓";
  if (node.misjudged) return "misjudged ✗";
  if (node.status === "pruned" || node.status === "dead") return "pruned ✗";
  return VERDICT_LABEL[node.verdict];
}

// Pick the color for the above-node label.
function nodeLabelColor(node: ThoughtNode): string {
  if (node.status === "solution") return "#2f7d4f";
  if (node.misjudged) return ACCENT;
  return SLATE;
}

export function TreeOfThoughtsSearch() {
  const [presetId, setPresetId] = useState("bfs-wide");
  const [state, dispatch] = useReducer(reducer, undefined, () => {
    const preset = getPreset("bfs-wide");
    return initialState(preset.params, preset.seed);
  });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [inspectId, setInspectId] = useState<number | null>(null);

  const { params, log, outcome, visited } = state;

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !state.done, 900 / speed, () =>
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
    setInspectId(null);
    dispatch({ kind: "reset", presetId });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setInspectId(null);
      setPresetId(id);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  const width = 1000;
  const height = 420;
  const placed = useMemo(() => layout(state, width, height), [state]);
  const positions = useMemo(() => {
    const map = new Map<number, Placed>();
    for (const item of placed) map.set(item.node.id, item);
    return map;
  }, [placed]);

  const inspected =
    inspectId === null ? null : (positions.get(inspectId)?.node ?? null);
  const misjudgedCount = state.nodes.filter((n) => n.misjudged).length;
  const prunedReachable = state.nodes.filter(
    (n) => n.status === "pruned" && n.reachable,
  ).length;

  const outcomeBadge =
    outcome === "solved"
      ? { text: "solved ✓", color: "#2f7d4f" }
      : outcome === "failed"
        ? { text: "failed", color: ACCENT }
        : { text: "searching", color: SLATE };

  const preset = PRESETS.find((p) => p.id === presetId);

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => (
          <Button
            key={p.id}
            type="button"
            size="sm"
            variant={presetId === p.id ? "default" : "outline"}
            aria-pressed={presetId === p.id}
            onClick={() => loadPreset(p.id)}
          >
            {p.label}
          </Button>
        ))}
      </div>
      <p className="text-muted-foreground text-sm">{preset?.blurb}</p>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          role="img"
          aria-label={`A tree of partial Game of 24 solutions explored by ${params.algorithm.toUpperCase()}. The search is ${outcome} after visiting ${visited} states.`}
        >
          {placed.map(({ node, x, y }) => {
            if (node.parentId === null) return null;
            const parent = positions.get(node.parentId);
            if (!parent) return null;
            const kept =
              node.status === "kept" ||
              node.status === "solution" ||
              node.status === "expanded";
            return (
              <line
                key={`edge-${node.id}`}
                x1={parent.x}
                y1={parent.y}
                x2={x}
                y2={y}
                stroke={kept ? ACCENT : SLATE}
                strokeWidth={kept ? 2 : 1}
                strokeOpacity={kept ? 0.8 : 0.25}
              />
            );
          })}

          {placed.map(({ node, x, y }) => {
            const isInspect = node.id === inspectId;
            const dimmed = node.status === "pruned" || node.status === "dead";
            return (
              <g
                key={`node-${node.id}`}
                role="button"
                tabIndex={0}
                aria-label={`State ${node.numbers.join(" ")}, verdict ${VERDICT_LABEL[node.verdict]}, ${node.status}. Click to inspect.`}
                className="cursor-pointer outline-none"
                onClick={() => setInspectId(node.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setInspectId(node.id);
                  }
                }}
                onFocus={() => setInspectId(node.id)}
              >
                <rect
                  x={x - 34}
                  y={y - 13}
                  width={68}
                  height={26}
                  rx={6}
                  fill={STATUS_FILL[node.status]}
                  fillOpacity={dimmed ? 0.3 : 1}
                  stroke={
                    isInspect
                      ? "var(--color-foreground)"
                      : node.misjudged
                        ? ACCENT
                        : "transparent"
                  }
                  strokeWidth={isInspect ? 2 : node.misjudged ? 2 : 0}
                  strokeDasharray={node.misjudged ? "4 2" : undefined}
                />
                <text
                  x={x}
                  y={y + 4}
                  textAnchor="middle"
                  fill={
                    node.status === "kept" || node.status === "solution"
                      ? "#fff"
                      : "var(--color-foreground)"
                  }
                  fillOpacity={dimmed ? 0.6 : 1}
                  style={{ fontSize: 11, fontWeight: 500 }}
                >
                  {node.numbers
                    .map((value) =>
                      Math.abs(value - Math.round(value)) < 1e-9
                        ? String(Math.round(value))
                        : value.toFixed(1),
                    )
                    .join(" ")}
                </text>
                {node.depth > 0 && (
                  <text
                    x={x}
                    y={y - 17}
                    textAnchor="middle"
                    fill={nodeLabelColor(node)}
                    style={{ fontSize: 8.5 }}
                  >
                    {nodeLabel(node)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Color legend — every color is paired with a word so meaning is never color-alone */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Color legend">
        <span className="text-muted-foreground flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-5 rounded-sm"
            style={{ backgroundColor: ACCENT }}
            aria-hidden="true"
          />
          kept by search
        </span>
        <span className="text-muted-foreground flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-5 rounded-sm"
            style={{ backgroundColor: "#2f7d4f" }}
            aria-hidden="true"
          />
          solution &#x2713;
        </span>
        <span className="text-muted-foreground flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-5 rounded-sm opacity-30"
            style={{ backgroundColor: SLATE }}
            aria-hidden="true"
          />
          pruned &#x2717;
        </span>
        <span className="text-muted-foreground flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-5 rounded-sm border-2 border-dashed"
            style={{ borderColor: ACCENT }}
            aria-hidden="true"
          />
          misjudged (reachable, called impossible)
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge style={{ backgroundColor: outcomeBadge.color, color: "#fff" }}>
            {outcomeBadge.text}
          </Badge>
          <Badge variant="outline">visited {visited}</Badge>
          <Badge variant="outline">{state.nodes.length} states</Badge>
          {misjudgedCount > 0 && (
            <Badge style={{ backgroundColor: ACCENT, color: "#fff" }}>
              {misjudgedCount} misjudged
            </Badge>
          )}
        </div>
        <div className="w-36">
          {/* Speed controls animation pace only — not a paper parameter */}
          <Slider
            label="Playback speed (animation pace only)"
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
        <SimPanel title="Search controls">
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="BFS"
              active={params.algorithm === "bfs"}
              onClick={() =>
                intervene({ type: "setAlgorithm", algorithm: "bfs" })
              }
            />
            <Toggle
              label="DFS"
              active={params.algorithm === "dfs"}
              onClick={() =>
                intervene({ type: "setAlgorithm", algorithm: "dfs" })
              }
            />
            <Toggle
              label="Honest evaluator"
              active={params.honest}
              onClick={() => intervene({ type: "toggleHonest" })}
            />
          </div>
          <div className="flex flex-wrap gap-1.5">
            {PUZZLES.map((puzzle) => (
              <Toggle
                key={puzzle.id}
                label={puzzle.label}
                active={params.puzzleId === puzzle.id}
                onClick={() => intervene({ type: "setPuzzle", id: puzzle.id })}
              />
            ))}
          </div>
          <Slider
            label="Beam width b — states kept per depth (BFS)"
            value={params.beam}
            min={1}
            max={6}
            step={1}
            display={String(params.beam)}
            onChange={(value) => intervene({ type: "setBeam", value })}
          />
          <Slider
            label="Prune threshold v_th — value below which DFS backtracks"
            value={params.threshold}
            min={0}
            max={0.95}
            step={0.05}
            display={params.threshold.toFixed(2)}
            onChange={(value) => intervene({ type: "setThreshold", value })}
          />
          <Slider
            label="Evaluator noise — chance it mislabels a reachable state (disabled when honest)"
            value={params.evalNoise}
            min={0}
            max={1}
            step={0.05}
            display={
              params.honest ? "off (honest)" : params.evalNoise.toFixed(2)
            }
            disabled={params.honest}
            onChange={(value) => intervene({ type: "setNoise", value })}
          />
          <p className="text-muted-foreground text-xs">
            {PUZZLES.find((p) => p.id === params.puzzleId)?.note}
          </p>
        </SimPanel>

        <SimPanel title="Selected state">
          {inspected ? (
            <div className="flex flex-col gap-2 text-xs">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">
                  numbers{" "}
                  {inspected.numbers
                    .map((v) =>
                      Math.abs(v - Math.round(v)) < 1e-9
                        ? String(Math.round(v))
                        : v.toFixed(2),
                    )
                    .join(", ")}
                </Badge>
                <Badge variant="outline">depth {inspected.depth}</Badge>
                <Badge variant="outline">{inspected.status}</Badge>
                <Badge
                  style={{
                    backgroundColor: inspected.reachable ? "#2f7d4f" : SLATE,
                    color: "#fff",
                  }}
                >
                  {inspected.reachable ? "24 reachable" : "24 unreachable"}
                </Badge>
              </div>
              <p className="text-muted-foreground font-mono">
                {inspected.depth === 0 ? "root puzzle" : inspected.thought}
                {" — verdict "}
                {VERDICT_LABEL[inspected.verdict]} (value{" "}
                {inspected.value.toFixed(3)})
                {inspected.misjudged
                  ? ", a confident mistake: reachable but called impossible"
                  : ""}
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    intervene({ type: "flipVerdict", nodeId: inspected.id })
                  }
                  disabled={inspected.status === "solution"}
                >
                  Flip verdict
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    intervene({ type: "forcePrune", nodeId: inspected.id })
                  }
                >
                  Force prune
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    intervene({ type: "forceKeep", nodeId: inspected.id })
                  }
                >
                  Force keep
                </Button>
              </div>
              <p className="text-muted-foreground">
                {childrenOf(state, inspected.id).length} children explored from
                here.
              </p>
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click any state in the tree to read its numbers, the equation that
              made it, the evaluator&apos;s verdict, and whether 24 is truly
              still reachable. Then flip its verdict or prune it and rerun.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Pruned but truly reachable: {prunedReachable}
            </span>
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step the search or poke a state to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.id}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The thought generator and the state evaluator stand in for the two jobs
        the paper gives the language model. Game of 24 is small enough that the{" "}
        <span style={{ color: SAGE }}>24 reachable</span> light is exact
        arithmetic, never a guess, so an honest evaluator always finds a route
        when one exists. The evaluator noise knob (active only when the honest
        toggle is off) is what turns the evaluator into a fallible model that
        prunes good branches, the failure the paper sees on crosswords. The run
        is deterministic: same preset, same search every time. This runs one
        puzzle with a handful of branches; the paper runs GPT-4 over 100 games.
      </p>
    </div>
  );
}
