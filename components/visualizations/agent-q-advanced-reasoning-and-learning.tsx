"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  blendedValue,
  pageOf,
  presetState,
  PRESETS,
  step,
  type Input,
  type Intervention,
  type SimState,
  type TreeNode,
} from "@/components/visualizations/agent-q-advanced-reasoning-and-learning-model";

const ACCENT = "#c2683f"; // clay orange: the goal / high value
const SAGE = "#6f8f6a"; // sage: low value / the safe-but-weak path
const DANGER = "#b04a3a"; // dusty red: a failure terminal or a poisoned critic

// The reducer carries a bounded stack of prior states so the reader can rewind.
// A tick or intervention pushes the old state; "reset" clears the stack; "back"
// pops to the previous state. Rewinding works because the model is deterministic,
// so a restored state replays forward identically.
interface History {
  current: SimState;
  past: SimState[];
}

const HISTORY_LIMIT = 80;

type ViewAction =
  | Input
  | { kind: "reset"; presetId: string }
  | { kind: "back" }
  | { kind: "restoreTo"; eventId: number };

function reducer(history: History, action: ViewAction): History {
  if (action.kind === "reset") {
    return { current: presetState(action.presetId), past: [] };
  }
  if (action.kind === "back") {
    const past = [...history.past];
    const previous = past.pop();
    if (previous === undefined) return history;
    return { current: previous, past };
  }
  if (action.kind === "restoreTo") {
    // Each event id was assigned in order, and each dispatch pushed exactly one
    // prior state, so the state whose log ends at this event id is the snapshot
    // to restore. Find it in the stack and make it current.
    const all = [...history.past, history.current];
    const index = all.findIndex(
      (s) => s.log.events[s.log.events.length - 1]?.id === action.eventId,
    );
    if (index < 0) return history;
    return { current: all[index]!, past: all.slice(0, index) };
  }
  const next = step(history.current, action);
  const past = [...history.past, history.current].slice(-HISTORY_LIMIT);
  return { current: next, past };
}

function init(presetId: string): History {
  return { current: presetState(presetId), past: [] };
}

type ClickMode = "inspect" | "poison" | "kill";

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
  step: stepBy,
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
        step={stepBy}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer"
        style={{ accentColor: ACCENT }}
      />
    </label>
  );
}

// Mix from sage (low value) through cream to clay (high value), so a node's fill
// reads its blended Q at a glance. Color is never the only cue: shape and the
// numeric label carry the same meaning for color blindness.
function valueColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  if (v >= 0.5) return ACCENT;
  return SAGE;
}

interface Laid {
  node: TreeNode;
  x: number;
  y: number;
  value: number;
}

// Lay the tree out by depth (rows) and breadth (columns within a depth), so the
// root sits at top and the booking path runs downward. Pure given the node list.
function layout(state: SimState, width: number, height: number): Laid[] {
  const byDepth = new Map<number, TreeNode[]>();
  for (const node of state.nodes) {
    const list = byDepth.get(node.depth) ?? [];
    list.push(node);
    byDepth.set(node.depth, list);
  }
  const maxDepth = Math.max(...state.nodes.map((n) => n.depth), 1);
  const rowGap = (height - 80) / (maxDepth + 1);
  const laid: Laid[] = [];
  for (const [depth, list] of [...byDepth.entries()].sort(
    (a, b) => a[0] - b[0],
  )) {
    const colGap = width / (list.length + 1);
    list.sort((a, b) => a.id - b.id);
    list.forEach((node, index) => {
      const page = pageOf(node.pageId);
      const value = page.terminal
        ? page.success
          ? 1
          : 0
        : Math.max(
            ...page.actions.map((_, i) =>
              blendedValue(node, i, state.params.alpha),
            ),
            0,
          );
      laid.push({
        node,
        x: colGap * (index + 1),
        y: 50 + rowGap * depth,
        value,
      });
    });
  }
  return laid;
}

export function AgentQSearch() {
  const [history, dispatch] = useReducer(reducer, "guided", init);
  const state = history.current;
  const canRewind = history.past.length > 0;
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("guided");
  const [clickMode, setClickMode] = useState<ClickMode>("inspect");
  const [inspectId, setInspectId] = useState<number | null>(0);

  const { params } = state;

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 850 / speed, () =>
    dispatch({ kind: "tick" }),
  );

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onBack = useCallback(() => {
    setPlaying(false);
    resetClock();
    dispatch({ kind: "back" });
  }, [resetClock]);
  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    setInspectId(0);
    dispatch({ kind: "reset", presetId });
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      setInspectId(0);
      dispatch({ kind: "reset", presetId: id });
    },
    [resetClock],
  );

  const width = 760;
  const height = 420;
  const laid = useMemo(() => layout(state, width, height), [state]);
  const posById = useMemo(() => {
    const map = new Map<number, Laid>();
    for (const l of laid) map.set(l.node.id, l);
    return map;
  }, [laid]);

  const selectedSet = useMemo(
    () => new Set(state.selectedPath),
    [state.selectedPath],
  );

  const onNodeClick = useCallback(
    (nodeId: number) => {
      if (clickMode === "poison") intervene({ type: "poisonNode", nodeId });
      else if (clickMode === "kill") intervene({ type: "killNode", nodeId });
      else setInspectId(nodeId);
    },
    [clickMode, intervene],
  );

  const successRate =
    state.iteration > 0 ? state.successes / state.iteration : 0;

  const inspected =
    inspectId === null
      ? null
      : (state.nodes.find((n) => n.id === inspectId) ?? null);
  const inspectedPage = inspected ? pageOf(inspected.pageId) : null;

  const preset = PRESETS.find((p) => p.id === presetId);

  return (
    <div className="not-prose my-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((item) => (
          <Button
            key={item.id}
            type="button"
            size="sm"
            variant={presetId === item.id ? "default" : "outline"}
            aria-pressed={presetId === item.id}
            onClick={() => loadPreset(item.id)}
          >
            {item.label}
          </Button>
        ))}
      </div>
      <p className="text-muted-foreground text-sm">{preset?.blurb}</p>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          role="img"
          aria-label={`Agent Q search tree over web pages after ${state.iteration} rollouts. ${state.successes} reached the booking. Click a node to inspect it.`}
        >
          {laid.map((l) => {
            const parentId = l.node.parentId;
            if (parentId === null) return null;
            const parent = posById.get(parentId);
            if (parent === undefined) return null;
            const onPath =
              selectedSet.has(l.node.id) && selectedSet.has(parentId);
            return (
              <line
                key={`edge-${l.node.id}`}
                x1={parent.x}
                y1={parent.y + 16}
                x2={l.x}
                y2={l.y - 16}
                stroke={onPath ? ACCENT : "currentColor"}
                strokeOpacity={onPath ? 0.9 : 0.18}
                strokeWidth={onPath ? 3 : 1.5}
                className="text-foreground"
              />
            );
          })}

          {laid.map((l) => {
            const page = pageOf(l.node.pageId);
            const isTerminal = page.terminal;
            const fill = isTerminal
              ? page.success
                ? ACCENT
                : DANGER
              : valueColor(l.value);
            const isInspected = l.node.id === inspectId;
            const r = l.node.id === state.rootId ? 18 : 15;
            return (
              <g
                key={`node-${l.node.id}`}
                role="button"
                tabIndex={0}
                aria-label={`${page.label}, value ${l.value.toFixed(2)}, visited ${l.node.visits} times`}
                className="cursor-pointer outline-none"
                onClick={() => onNodeClick(l.node.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onNodeClick(l.node.id);
                  }
                }}
              >
                {isTerminal ? (
                  <rect
                    x={l.x - r}
                    y={l.y - r}
                    width={r * 2}
                    height={r * 2}
                    rx={4}
                    fill={fill}
                    fillOpacity={l.node.dead ? 0.25 : 0.92}
                    stroke={isInspected ? "#000" : "transparent"}
                    strokeWidth={2}
                  />
                ) : (
                  <circle
                    cx={l.x}
                    cy={l.y}
                    r={r}
                    fill={fill}
                    fillOpacity={l.node.dead ? 0.2 : 0.9}
                    stroke={
                      isInspected
                        ? "#000"
                        : l.node.poisoned
                          ? DANGER
                          : "transparent"
                    }
                    strokeWidth={l.node.poisoned ? 3 : 2}
                    strokeDasharray={l.node.poisoned ? "3 2" : undefined}
                  />
                )}
                {l.node.visits > 0 && !isTerminal && (
                  <text
                    x={l.x}
                    y={l.y + 4}
                    textAnchor="middle"
                    fill="#fff"
                    style={{ fontSize: 11, fontWeight: 600 }}
                  >
                    {l.node.visits}
                  </text>
                )}
                {isTerminal && (
                  <text
                    x={l.x}
                    y={l.y + 4}
                    textAnchor="middle"
                    fill="#fff"
                    style={{ fontSize: 11, fontWeight: 700 }}
                  >
                    {page.success ? "win" : "fail"}
                  </text>
                )}
                <text
                  x={l.x}
                  y={l.y + r + 13}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{ fontSize: 10 }}
                >
                  {page.label}
                </text>
                {l.node.dead && (
                  <text
                    x={l.x}
                    y={l.y - r - 5}
                    textAnchor="middle"
                    fill={DANGER}
                    style={{ fontSize: 9 }}
                  >
                    killed
                  </text>
                )}
                {l.node.poisoned && !l.node.dead && (
                  <text
                    x={l.x}
                    y={l.y - r - 5}
                    textAnchor="middle"
                    fill={DANGER}
                    style={{ fontSize: 9 }}
                  >
                    critic flipped
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <PlayPauseStepControls
            playing={playing}
            onPlayPause={onPlayPause}
            onStep={onStep}
            onReset={onReset}
          />
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={onBack}
            disabled={!canRewind || playing}
            aria-label="Step back one rollout"
          >
            Back
          </Button>
        </div>
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

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">Click a node to:</span>
        {(["inspect", "poison", "kill"] as ClickMode[]).map((mode) => (
          <Button
            key={mode}
            type="button"
            size="sm"
            variant={clickMode === mode ? "default" : "outline"}
            aria-pressed={clickMode === mode}
            onClick={() => setClickMode(mode)}
          >
            {mode === "inspect"
              ? "Inspect"
              : mode === "poison"
                ? "Flip its critic"
                : "Kill branch"}
          </Button>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Search knobs">
          <Slider
            label="Exploration c_exp (0 = greedy)"
            value={params.cExp}
            min={0}
            max={3}
            step={0.1}
            display={params.cExp.toFixed(1)}
            onChange={(value) => intervene({ type: "setCExp", value })}
          />
          <Slider
            label="alpha: trust rollouts vs critic"
            value={params.alpha}
            min={0}
            max={1}
            step={0.05}
            display={`${params.alpha.toFixed(2)} (1=rollouts)`}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          <Slider
            label="DPO pair threshold"
            value={params.threshold}
            min={0.05}
            max={0.9}
            step={0.05}
            display={params.threshold.toFixed(2)}
            onChange={(value) => intervene({ type: "setThreshold", value })}
          />
          <Slider
            label="Rollout noise"
            value={params.rolloutNoise}
            min={0}
            max={1}
            step={0.05}
            display={params.rolloutNoise.toFixed(2)}
            onChange={(value) => intervene({ type: "setRolloutNoise", value })}
          />
          <Slider
            label="Search budget (rollouts)"
            value={params.budget}
            min={4}
            max={40}
            step={2}
            display={String(params.budget)}
            onChange={(value) => intervene({ type: "setBudget", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the search">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              rollout {state.iteration}/{params.budget}
            </Badge>
            <Badge variant="outline">
              booked {state.successes} ({Math.round(successRate * 100)}%)
            </Badge>
            <Badge variant="outline">{state.nodes.length} nodes</Badge>
            <Badge variant="outline">{state.prefPairs.length} DPO pairs</Badge>
          </div>

          {inspected && inspectedPage ? (
            <div className="ring-foreground/10 rounded-lg p-2 text-xs ring-1">
              <p className="text-foreground font-medium">
                {inspectedPage.label}
              </p>
              <p className="text-muted-foreground">{inspectedPage.note}</p>
              {!inspectedPage.terminal && (
                <table className="mt-1 w-full text-left font-mono">
                  <thead className="text-muted-foreground">
                    <tr>
                      <th className="py-0.5 font-medium">action</th>
                      <th className="py-0.5 text-right font-medium">critic</th>
                      <th className="py-0.5 text-right font-medium">MCTS</th>
                      <th className="py-0.5 text-right font-medium">blend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspectedPage.actions.map((action, i) => (
                      <tr key={action.id} className="text-muted-foreground">
                        <td className="py-0.5">{action.label}</td>
                        <td className="py-0.5 text-right">
                          {(inspected.poisoned
                            ? 1 - action.criticRank
                            : action.criticRank
                          ).toFixed(2)}
                        </td>
                        <td className="py-0.5 text-right">
                          {(inspected.actionValue[i] ?? 0).toFixed(2)}
                        </td>
                        <td className="text-foreground py-0.5 text-right">
                          {blendedValue(inspected, i, params.alpha).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Click a node to read its page and its per-action critic, MCTS, and
              blended values.
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Preference pairs for DPO
            </span>
            <div className="ring-foreground/10 max-h-24 overflow-y-auto rounded-lg text-xs ring-1">
              {state.prefPairs.length === 0 ? (
                <p className="text-muted-foreground px-2 py-1.5">
                  No sibling gap clears the threshold yet. Run more rollouts.
                </p>
              ) : (
                <ul>
                  {state.prefPairs.map((pair, i) => (
                    <li
                      key={`${pair.nodeId}-${i}`}
                      className="text-muted-foreground px-2 py-1"
                    >
                      <span className="text-foreground">{pair.pageLabel}</span>:
                      prefer &quot;{pair.preferred}&quot; over &quot;
                      {pair.rejected}&quot; (gap {pair.gap.toFixed(2)})
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={state.log.events}
              emptyHint="Step the search or poke a node to start the log."
              onRestore={(event) => {
                setPlaying(false);
                resetClock();
                dispatch({ kind: "restoreTo", eventId: event.id });
              }}
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs one base LLM as both the actor that proposes actions and the
        critic that ranks them, exactly as in the paper, over a fixed 8-page
        booking task standing in for OpenTable. The per-action critic ranks, the
        actor proposals, and each action&apos;s true reach-the-goal chance are
        hand-set so the search is legible; a trained agent learns these from
        data. The DPO fine-tuning step itself is summarised by the preference
        pairs the search produces, not run as gradient descent here.
      </p>
    </div>
  );
}
