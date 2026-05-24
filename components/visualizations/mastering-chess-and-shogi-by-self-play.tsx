"use client";

import {
  useCallback,
  useMemo,
  useReducer,
  useState,
  type ReactNode,
} from "react";

import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  getNode,
  getScenario,
  initialState,
  meanValue,
  playsBestMove,
  rootMoveStats,
  SCENARIOS,
  selectedMove,
  step,
  type GameTree,
  type Input,
  type Intervention,
  type SimState,
} from "@/components/visualizations/mastering-chess-and-shogi-by-self-play-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#7d8c6a";

function reducer(state: SimState, input: Input): SimState {
  return step(state, input);
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

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1">
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
  );
}

// Lay the fixed game tree out by depth (rows) and breadth (columns within a
// depth), so the same scenario always draws in the same place and the reader can
// watch visits accumulate node by node.
interface Layout {
  positions: Record<number, { x: number; y: number }>;
  width: number;
  height: number;
}

function layoutTree(tree: GameTree): Layout {
  const byDepth: Record<number, number[]> = {};
  for (const node of tree.nodes) {
    (byDepth[node.depth] ??= []).push(node.id);
  }
  const depths = Object.keys(byDepth)
    .map(Number)
    .sort((a, b) => a - b);
  const maxDepth = depths.length - 1;
  const width = 760;
  const rowGap = 130;
  const top = 46;
  const positions: Record<number, { x: number; y: number }> = {};
  for (const depth of depths) {
    const row = byDepth[depth] ?? [];
    const slot = width / (row.length + 1);
    row.forEach((id, index) => {
      positions[id] = { x: slot * (index + 1), y: top + depth * rowGap };
    });
  }
  return { positions, width, height: top + maxDepth * rowGap + 60 };
}

export function MonteCarloTreeSearch() {
  const [scenarioId, setScenarioId] = useState("wrong-net");
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("wrong-net"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.5);
  const [inspect, setInspect] = useState<number | null>(null);

  const scenario = getScenario(state.scenarioId);
  const tree = scenario.tree;
  const layout = useMemo(() => layoutTree(tree), [tree]);
  const moves = rootMoveStats(state);
  const chosen = selectedMove(state);
  const correct = playsBestMove(state);
  const onBudget = state.completed >= state.params.simBudget;
  const maxVisits = Math.max(1, ...moves.map((move) => move.visits));

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing && !onBudget, 900 / speed, () =>
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
    intervene({ type: "loadScenario", id: scenarioId });
  }, [intervene, resetClock, scenarioId]);

  const loadScenario = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setScenarioId(id);
      setInspect(null);
      intervene({ type: "loadScenario", id });
    },
    [intervene, resetClock],
  );

  const activePath = new Set(state.path);
  const inspected = inspect === null ? null : getNode(tree, inspect);
  const inspectedStat = inspect === null ? null : state.stats[inspect];

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {SCENARIOS.map((item) => (
          <Button
            key={item.id}
            type="button"
            size="sm"
            variant={scenarioId === item.id ? "default" : "outline"}
            aria-pressed={scenarioId === item.id}
            onClick={() => loadScenario(item.id)}
          >
            {item.label}
          </Button>
        ))}
      </div>
      <p className="text-muted-foreground text-sm">{scenario.blurb}</p>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="outline">
          sim {state.completed} / {state.params.simBudget}
        </Badge>
        <Badge variant="secondary">plays {chosen ? chosen.label : "-"}</Badge>
        <Badge
          variant="outline"
          style={{
            borderColor: correct ? SAGE : ACCENT,
            color: correct ? SAGE : ACCENT,
          }}
        >
          {correct ? "plays the true best move" : "not yet the best move"}
        </Badge>
      </div>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="w-full"
          role="img"
          aria-label={`Monte-Carlo search tree for the ${scenario.label} scenario after ${state.completed} simulations. The search currently plays move ${chosen ? chosen.label : "none"}.`}
        >
          {tree.nodes.map((node) => {
            if (node.parent === null) return null;
            const a = layout.positions[node.parent];
            const b = layout.positions[node.id];
            if (!a || !b) return null;
            const onPath =
              activePath.has(node.id) && activePath.has(node.parent);
            const visits = state.stats[node.id]?.visits ?? 0;
            return (
              <line
                key={`edge-${node.id}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={onPath ? ACCENT : "var(--color-border)"}
                strokeWidth={onPath ? 3 : 1 + Math.min(visits, 30) * 0.18}
                strokeOpacity={onPath ? 0.95 : 0.45}
              />
            );
          })}

          {tree.nodes.map((node) => {
            const pos = layout.positions[node.id];
            if (!pos) return null;
            const stat = state.stats[node.id] ?? {
              visits: 0,
              valueSum: 0,
              expanded: false,
            };
            const onPath = activePath.has(node.id);
            const isLeafNode = node.children.length === 0;
            const radius = 13 + Math.min(stat.visits, 40) * 0.55;
            const q = -meanValue(stat);
            const fill =
              stat.visits === 0
                ? "var(--color-muted)"
                : q > 0.05
                  ? ACCENT
                  : q < -0.05
                    ? SAGE
                    : ACCENT_SOFT;
            return (
              <g
                key={`node-${node.id}`}
                role="button"
                tabIndex={0}
                aria-label={`Inspect node ${node.label}, ${stat.visits} visits`}
                className="cursor-pointer outline-none"
                onClick={() => setInspect(node.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setInspect(node.id);
                  }
                }}
                onMouseEnter={() => setInspect(node.id)}
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={radius}
                  fill={fill}
                  fillOpacity={stat.visits === 0 ? 0.5 : 0.92}
                  stroke={
                    onPath
                      ? ACCENT
                      : node.id === inspect
                        ? "var(--color-foreground)"
                        : "transparent"
                  }
                  strokeWidth={onPath ? 3 : 2}
                  strokeDasharray={isLeafNode ? "3 2" : undefined}
                />
                <text
                  x={pos.x}
                  y={pos.y + 4}
                  textAnchor="middle"
                  fill={
                    stat.visits === 0 ? "var(--color-muted-foreground)" : "#fff"
                  }
                  style={{ fontSize: 12, fontWeight: 600 }}
                >
                  {node.label}
                </text>
                {stat.visits > 0 && node.parent !== null && (
                  <text
                    x={pos.x}
                    y={pos.y - radius - 5}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 10 }}
                  >
                    N={stat.visits}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
          disabled={onBudget && !playing && state.completed === 0}
        />
        <div className="w-40">
          <Slider
            label="Speed"
            value={speed}
            min={0.5}
            max={4}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>
      {onBudget && (
        <p className="text-muted-foreground text-xs">
          Budget reached. Raise the simulation budget or reset to keep
          searching.
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Search controls">
          <Slider
            label="Exploration c_puct"
            value={state.params.cPuct}
            min={0}
            max={5}
            step={0.1}
            display={state.params.cPuct.toFixed(1)}
            onChange={(value) => intervene({ type: "setCPuct", value })}
          />
          <Slider
            label="Root noise (Dirichlet mix)"
            value={state.params.rootNoise}
            min={0}
            max={1}
            step={0.05}
            display={
              state.params.rootNoise === 0
                ? "off"
                : state.params.rootNoise.toFixed(2)
            }
            onChange={(value) => intervene({ type: "setRootNoise", value })}
          />
          <Slider
            label="Simulation budget"
            value={state.params.simBudget}
            min={4}
            max={120}
            step={4}
            display={String(state.params.simBudget)}
            onChange={(value) => intervene({ type: "setSimBudget", value })}
          />
          <Slider
            label="Seed (sets the noise stream)"
            value={state.params.seed}
            min={0}
            max={20}
            step={1}
            display={String(state.params.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
          <p className="text-muted-foreground text-xs">
            The seed only changes the run when root noise is on, and only takes
            effect before the first simulation.
          </p>
        </Panel>

        <Panel title="Root moves and the inside">
          <div className="flex flex-col gap-1.5">
            {moves.map((move) => {
              const isChosen = chosen?.id === move.id;
              const isBest = move.id === scenario.bestMove;
              return (
                <div key={move.id} className="flex items-center gap-2 text-xs">
                  <span className="w-5 font-mono">{move.label}</span>
                  <div className="bg-foreground/5 relative h-4 flex-1 overflow-hidden rounded">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${(move.visits / maxVisits) * 100}%`,
                        backgroundColor: isChosen ? ACCENT : ACCENT_SOFT,
                      }}
                    />
                  </div>
                  <span className="text-muted-foreground w-8 text-right font-mono">
                    {move.visits}
                  </span>
                  <span
                    className="w-4 text-center"
                    aria-label={isBest ? "true best move" : undefined}
                    title={isBest ? "true best move" : undefined}
                    style={{ color: isBest ? SAGE : "transparent" }}
                  >
                    {isBest ? "✓" : "·"}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="ring-foreground/10 rounded-lg p-2 text-xs ring-1">
            {inspected && inspectedStat ? (
              <div className="flex flex-col gap-1 font-mono">
                <span className="text-foreground font-sans font-medium">
                  node {inspected.label}
                </span>
                <span className="text-muted-foreground">
                  prior P = {inspected.prior.toFixed(2)}
                </span>
                <span className="text-muted-foreground">
                  net value v = {inspected.netValue.toFixed(2)}
                </span>
                <span className="text-muted-foreground">
                  visits N = {inspectedStat.visits}
                </span>
                <span className="text-muted-foreground">
                  mean value Q = {(-meanValue(inspectedStat)).toFixed(2)}
                </span>
                {inspected.children.length === 0 && (
                  <span className="text-muted-foreground">
                    true outcome = {inspected.trueValue}
                  </span>
                )}
              </div>
            ) : (
              <span className="text-muted-foreground">
                Hover or click a node to read its prior, value, and visits.
              </span>
            )}
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <div className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1">
              {state.events.length === 0 ? (
                <p className="text-muted-foreground px-2 py-1.5 text-xs">
                  Step or play to start the log. Click an entry to rewind there.
                </p>
              ) : (
                <ul className="text-xs">
                  {[...state.events].reverse().map((event) => (
                    <li key={event.id}>
                      <button
                        type="button"
                        className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                        onClick={() => {
                          setPlaying(false);
                          resetClock();
                          intervene({
                            type: "restore",
                            snapshot: event.snapshot,
                            label: `rewind to sim ${event.sim}`,
                          });
                        }}
                      >
                        {event.id}. {event.label}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs the paper&apos;s PUCT search over a fixed nine-node game tree,
        a stand-in for chess where the real tree holds millions of positions.
        The network&apos;s prior P and value v are hand-set per node so the
        patterns are legible, where AlphaZero learns those numbers from
        self-play. The search reads the network&apos;s value at each leaf, not
        the true outcome, so a wrong, under-explored net plays the wrong move
        exactly as it would in the full game.
      </p>
    </div>
  );
}
