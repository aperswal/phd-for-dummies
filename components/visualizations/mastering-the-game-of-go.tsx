"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  chosenMove,
  initialState,
  isCorrect,
  minimaxValue,
  MOVE_NAMES,
  PRESETS,
  rootValueError,
  step,
  totalRootVisits,
  trueBestMove,
  type Intervention,
  type SimState,
  type TreeNode,
} from "@/components/visualizations/mastering-the-game-of-go-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
const INK = "#2f2a26";
// Neutral fill for nodes — keeps ACCENT reserved for error/warning states
// (wrong-move label, last-path edge) so the two meanings never collide.
const NODE_NEUTRAL = "var(--color-muted-foreground)";

type ClickMode = "inspect" | "boost" | "lie" | "prune";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset")
    return initialState(action.presetId, state.baseSeed);
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

interface Placed {
  node: TreeNode;
  x: number;
  y: number;
}

// Lay the fixed-shape tree out by depth (rows) and slot order (columns within a
// row), so the picture is stable as the tree fills in. Unexpanded subtrees just
// have fewer placed nodes.
function layout(state: SimState, width: number): Placed[] {
  const byDepth = new Map<number, TreeNode[]>();
  for (const node of state.nodes) {
    const row = byDepth.get(node.depth) ?? [];
    row.push(node);
    byDepth.set(node.depth, row);
  }
  const placed: Placed[] = [];
  const topPad = 40;
  const rowHeight = 110;
  for (const [depth, row] of byDepth) {
    // Keep children grouped under their parent by sorting on the slot path.
    row.sort((a, b) => orderKey(state, a) - orderKey(state, b));
    const slot = width / (row.length + 1);
    row.forEach((node, index) => {
      placed.push({
        node,
        x: slot * (index + 1),
        y: topPad + depth * rowHeight,
      });
    });
  }
  return placed;
}

// A monotonic key that groups siblings together left to right.
function orderKey(state: SimState, node: TreeNode): number {
  const slots: number[] = [];
  let current: TreeNode | undefined = node;
  while (current && current.parent !== null) {
    slots.unshift(current.childSlot);
    current = state.nodes[current.parent];
  }
  return slots.reduce((acc, s) => acc * 10 + s + 1, 0);
}

function mean(node: TreeNode): number {
  return node.visits === 0 ? 0 : node.totalValue / node.visits;
}

function nodeName(state: SimState, node: TreeNode): string {
  if (node.parent === null) return "root";
  if (node.parent === state.root) {
    return MOVE_NAMES[node.childSlot] ?? `move ${node.childSlot}`;
  }
  const parent = state.nodes[node.parent];
  const parentName =
    parent && parent.parent === state.root
      ? (MOVE_NAMES[parent.childSlot] ?? `move ${parent.childSlot}`)
      : "node";
  return `${parentName} → reply ${node.childSlot + 1}`;
}

export function GoTreeSearch() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("honest"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const [presetId, setPresetId] = useState("honest");
  const [clickMode, setClickMode] = useState<ClickMode>("inspect");
  const [inspect, setInspect] = useState<number | null>(null);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 220 / speed, () =>
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
      setInspect(null);
      dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
    },
    [resetClock],
  );

  const onNodeClick = useCallback(
    (node: TreeNode) => {
      if (node.parent === null) {
        setInspect(node.id);
        return;
      }
      if (clickMode === "boost") {
        if (node.priorBias > 1)
          intervene({ type: "resetPrior", node: node.id });
        else intervene({ type: "boostPrior", node: node.id });
      } else if (clickMode === "lie") {
        if (node.valueLie !== null)
          intervene({ type: "clearLie", node: node.id });
        else intervene({ type: "lieValue", node: node.id, value: 0.9 });
      } else if (clickMode === "prune") {
        if (node.pruned) intervene({ type: "unprune", node: node.id });
        else intervene({ type: "prune", node: node.id });
      } else {
        setInspect(node.id);
      }
    },
    [clickMode, intervene],
  );

  const width = 920;
  const placed = useMemo(() => layout(state, width), [state]);
  const positionOf = useMemo(() => {
    const map = new Map<number, Placed>();
    for (const p of placed) map.set(p.node.id, p);
    return map;
  }, [placed]);

  const chosen = chosenMove(state);
  const truth = trueBestMove(state);
  const correct = isCorrect(state);
  const valueErr = rootValueError(state);
  const totalVisits = totalRootVisits(state);
  const lastPath = new Set(state.last?.path ?? []);

  const maxVisits = Math.max(1, ...state.nodes.map((n) => n.visits));
  const radiusFor = (node: TreeNode) => {
    if (node.parent === null) return 26;
    return 12 + 16 * Math.sqrt(node.visits / maxVisits);
  };

  const height = 40 + 110 * 2 + 60;
  const inspected = inspect !== null ? (state.nodes[inspect] ?? null) : null;

  const modes: { id: ClickMode; label: string }[] = [
    { id: "inspect", label: "Inspect" },
    { id: "boost", label: "Boost prior" },
    { id: "lie", label: "Lie value" },
    { id: "prune", label: "Prune" },
  ];

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
          aria-label={`A search tree of ${state.nodes.length} positions after ${state.simulations} simulations. The most-visited root move is ${chosen !== null ? nodeName(state, state.nodes[chosen]!) : "none yet"}, and the search has played ${state.simulations} simulations.`}
        >
          {placed.map(({ node, x, y }) => {
            if (node.parent === null) return null;
            const parent = positionOf.get(node.parent);
            if (!parent) return null;
            const onLastPath =
              lastPath.has(node.id) && lastPath.has(node.parent);
            return (
              <line
                key={`edge-${node.id}`}
                x1={parent.x}
                y1={parent.y}
                x2={x}
                y2={y}
                stroke={onLastPath ? ACCENT : INK}
                strokeOpacity={node.pruned ? 0.15 : onLastPath ? 0.9 : 0.25}
                strokeWidth={onLastPath ? 3 : 1 + 2 * (node.visits / maxVisits)}
              />
            );
          })}

          {placed.map(({ node, x, y }) => {
            const r = radiusFor(node);
            const isChosen = node.id === chosen;
            const isTruth = node.id === truth;
            const q = mean(node);
            const lying = node.valueLie !== null;
            const boosted = node.priorBias > 1;
            const fill = node.pruned
              ? "var(--color-muted)"
              : lying
                ? SAGE
                : node.parent === null
                  ? "var(--color-foreground)"
                  : NODE_NEUTRAL;
            const opacity = node.pruned
              ? 0.3
              : node.parent === null
                ? 1
                : 0.25 + 0.6 * (node.visits / maxVisits);
            return (
              <g
                key={`node-${node.id}`}
                role="button"
                tabIndex={0}
                aria-label={`${nodeName(state, node)}, ${node.visits} visits, mean value ${q.toFixed(2)}`}
                aria-pressed={node.id === inspect}
                className="cursor-pointer outline-none"
                onClick={() => onNodeClick(node)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onNodeClick(node);
                  }
                }}
                onMouseEnter={() => setInspect(node.id)}
                onFocus={() => setInspect(node.id)}
              >
                <circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill={fill}
                  fillOpacity={opacity}
                  stroke={
                    node.id === inspect
                      ? ACCENT
                      : isTruth
                        ? SAGE
                        : "var(--color-border)"
                  }
                  strokeWidth={node.id === inspect ? 3 : isTruth ? 2.5 : 1}
                  strokeDasharray={isTruth && !isChosen ? "4 3" : undefined}
                />
                {isChosen && node.parent !== null && (
                  <circle
                    cx={x}
                    cy={y}
                    r={r + 5}
                    fill="none"
                    stroke={ACCENT}
                    strokeWidth={2}
                  />
                )}
                {node.parent !== null && node.visits > 0 && (
                  <text
                    x={x}
                    y={y + 4}
                    textAnchor="middle"
                    fill="var(--color-background)"
                    style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                  >
                    {node.visits}
                  </text>
                )}
                {node.parent === state.root && (
                  <text
                    x={x}
                    y={y + r + 16}
                    textAnchor="middle"
                    className="fill-foreground"
                    style={{ fontSize: 12, fontWeight: 500 }}
                  >
                    {MOVE_NAMES[node.childSlot]}
                  </text>
                )}
                {(lying || boosted) && node.parent !== null && (
                  <text
                    x={x}
                    y={y - r - 5}
                    textAnchor="middle"
                    fill={lying ? SAGE : ACCENT}
                    style={{ fontSize: 10, fontWeight: 600 }}
                  >
                    {lying ? "lie" : "boost"}
                  </text>
                )}
              </g>
            );
          })}

          <text
            x={14}
            y={20}
            className="fill-muted-foreground"
            style={{ fontSize: 12 }}
          >
            root position
          </text>
          <text
            x={width - 14}
            y={height - 14}
            textAnchor="end"
            fill={correct ? SAGE : ACCENT}
            style={{ fontSize: 13, fontWeight: 600 }}
          >
            {chosen !== null
              ? `plays ${MOVE_NAMES[state.nodes[chosen]!.childSlot]} · ${correct ? "best move" : "wrong move"}`
              : "no move yet"}
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground text-xs">Click a node to:</span>
        {modes.map((m) => (
          <Toggle
            key={m.id}
            label={m.label}
            active={clickMode === m.id}
            onClick={() => setClickMode(m.id)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="w-48">
          <Slider
            label="Playback pace — faster means more simulations per second (does not affect search logic)"
            value={speed}
            min={1}
            max={16}
            step={1}
            display={`${speed}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Search knobs">
          <Slider
            label="lambda — how much rollout outcome z mixes into V(s_L) = (1−λ)v_θ + λz; 0 = value only, 1 = rollout only"
            value={state.params.lambda}
            min={0}
            max={1}
            step={0.05}
            display={
              state.params.lambda === 0
                ? "value only"
                : state.params.lambda === 1
                  ? "rollout only"
                  : state.params.lambda.toFixed(2)
            }
            onChange={(value) => intervene({ type: "setLambda", value })}
          />
          <Slider
            label="c_puct — exploration constant; 0 = ride the prior, higher = check more alternatives before trusting it"
            value={state.params.cPuct}
            min={0}
            max={3}
            step={0.05}
            display={state.params.cPuct.toFixed(2)}
            onChange={(value) => intervene({ type: "setCPuct", value })}
          />
          <Slider
            label="rollout noise — spread on the fast playout (0 = exact leaf value, 1 = very noisy single sample)"
            value={state.params.rolloutNoise}
            min={0}
            max={1}
            step={0.05}
            display={state.params.rolloutNoise.toFixed(2)}
            onChange={(value) => intervene({ type: "setRolloutNoise", value })}
          />
          <Slider
            label="value-net error — how inaccurate the value network is (0 = perfect, 1 = highly noisy)"
            value={state.params.valueNoise}
            min={0}
            max={1}
            step={0.05}
            display={state.params.valueNoise.toFixed(2)}
            onChange={(value) => intervene({ type: "setValueNoise", value })}
          />
        </SimPanel>

        <SimPanel title="What the search has found">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{state.simulations} simulations</Badge>
            <Badge variant="outline">{totalVisits} root visits</Badge>
            <Badge variant="outline">{state.nodes.length} nodes</Badge>
            <Badge
              variant={correct ? "default" : "outline"}
              style={
                correct
                  ? { backgroundColor: SAGE, color: "#fff" }
                  : { color: ACCENT }
              }
            >
              {correct ? "plays the best move" : "plays a worse move"}
            </Badge>
            <Badge variant="outline">value gap {valueErr.toFixed(2)}</Badge>
          </div>

          {inspected ? (
            <div className="ring-foreground/10 rounded-lg ring-1">
              <div className="text-muted-foreground flex items-center justify-between px-2 py-1 text-xs">
                <span>{nodeName(state, inspected)}</span>
                <span>
                  true value {minimaxValue(state, inspected.id).toFixed(2)}
                </span>
              </div>
              <table className="w-full text-left font-mono text-xs">
                <tbody>
                  <tr>
                    <td className="text-muted-foreground px-2 py-1">prior P</td>
                    <td className="px-2 py-1 text-right">
                      {(inspected.prior * inspected.priorBias).toFixed(2)}
                      {inspected.priorBias > 1 ? " (boosted)" : ""}
                    </td>
                  </tr>
                  <tr>
                    <td className="text-muted-foreground px-2 py-1">
                      visits N
                    </td>
                    <td className="px-2 py-1 text-right">{inspected.visits}</td>
                  </tr>
                  <tr>
                    <td className="text-muted-foreground px-2 py-1">
                      mean value Q
                    </td>
                    <td className="px-2 py-1 text-right">
                      {mean(inspected).toFixed(3)}
                    </td>
                  </tr>
                  <tr>
                    <td className="text-muted-foreground px-2 py-1">
                      value-net read
                    </td>
                    <td className="px-2 py-1 text-right">
                      {inspected.valueLie !== null
                        ? `${inspected.valueLie.toFixed(2)} (lie)`
                        : "honest"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Hover or click a node to read its prior, visit count, and value.
              Switch the click mode above to boost a move&apos;s prior, make the
              value net lie about a node, or prune a branch.
            </p>
          )}

          {state.last && (
            <p className="text-muted-foreground text-xs">
              last simulation reached{" "}
              <span className="text-foreground font-mono">
                {nodeName(state, state.nodes[state.last.leaf]!)}
              </span>
              , value net said{" "}
              <span className="text-foreground font-mono">
                {state.last.vTheta.toFixed(2)}
              </span>
              , rollout said{" "}
              <span className="text-foreground font-mono">
                {state.last.rollout.toFixed(2)}
              </span>
              , backed up{" "}
              <span className="text-foreground font-mono">
                {state.last.evaluation.toFixed(2)}
              </span>
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Flip a knob, boost a prior, or make the value net lie to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to sim ${event.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        Selection here is exactly the paper&apos;s rule, argmax of Q(s,a) +
        c_puct&middot;P(s,a)&middot;sqrt(&Sigma;N)/(1+N), and a leaf is scored
        by V = (1&minus;&lambda;)v&theta; + &lambda;z, the value-net and rollout
        blend. This runs a fixed 3-by-3 toy tree with hand-set true values
        standing in for one Go position; the real AlphaGo searches a tree with
        roughly 250 moves per position and uses 13-layer networks trained on 30
        million positions. The move played is always the most-visited root
        child. The run is fully deterministic: a fixed seed is used so the same
        preset and knob positions always reproduce the same sequence of
        simulations.
      </p>
    </div>
  );
}
