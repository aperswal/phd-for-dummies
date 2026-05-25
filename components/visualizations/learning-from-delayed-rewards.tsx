"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { at } from "@/lib/simulation/array";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  ACTIONS,
  convergence,
  DOWN,
  greedyPolicyAction,
  initialState,
  isPenalty,
  isWall,
  LEFT,
  optimalValues,
  PRESETS,
  RIGHT,
  rowCol,
  step,
  UP,
  valueOf,
  type Behavior,
  type Intervention,
  type SimState,
} from "@/components/visualizations/learning-from-delayed-rewards-model";

// Clay-orange: the goal's "glow" — positive value flowing back from the reward.
// Amber-warm: penalty cells — a distinct warning hue that doesn't collide with ACCENT.
// SAGE is reserved for the "greedy policy optimal" success badge only.

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
// Penalty cells get a warm amber distinct from ACCENT so hazard ≠ "good value".
const PENALTY_FILL = "#a0743a";
const WALL = "#2f2a26";

type ClickMode = "inspect" | "wall" | "goal" | "teleport";
const ACTION_NAMES: Record<number, string> = {
  [UP]: "up",
  [RIGHT]: "right",
  [DOWN]: "down",
  [LEFT]: "left",
};
// Indexed by action; the on-screen direction of the greedy-move arrow.
const ARROW = [
  { dx: 0, dy: -1 },
  { dx: 1, dy: 0 },
  { dx: 0, dy: 1 },
  { dx: -1, dy: 0 },
] as const;

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
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

export function QLearningGrid() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("backup"),
  );
  const [playing, setPlaying] = useState(false);
  const [presetId, setPresetId] = useState("backup");
  const [clickMode, setClickMode] = useState<ClickMode>("inspect");
  const [inspect, setInspect] = useState<number | null>(null);

  const { world, params, q, visits, agent, log } = state;
  const star = useMemo(
    () => optimalValues(world, params),
    // V* depends on the world and on the reward/discount params, not on Q.
    [world, params],
  );
  const score = useMemo(() => convergence(state), [state]);

  const cells = world.width * world.height;
  const values = useMemo(() => {
    const list: number[] = [];
    for (let cell = 0; cell < cells; cell++) {
      list.push(isWall(world, cell) ? 0 : valueOf(q, cell));
    }
    return list;
  }, [q, world, cells]);

  // Colour scale anchored at 0, so orange always means real value flowing back
  // from the goal. Cells at or below 0 (only step costs seen so far) stay faint.
  // A relative min-to-max scale would instead light up the untouched zero-value
  // cells brightest before the goal is ever reached, since 0 beats a negative.
  const finite = values.filter((_, cell) => !isWall(world, cell));
  const vmax = Math.max(...finite, 1e-3);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Fixed playback pace of ~5 steps/s — fast enough to see learning unfold,
  // slow enough that each Q-update is legible. Speed is not a paper concept.
  const resetClock = useFixedTimestep(playing, 200, () =>
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

  const onCellClick = useCallback(
    (cell: number) => {
      if (clickMode === "wall") intervene({ type: "toggleWall", cell });
      else if (clickMode === "goal") intervene({ type: "setGoal", cell });
      else if (clickMode === "teleport") intervene({ type: "teleport", cell });
      else setInspect(cell);
    },
    [clickMode, intervene],
  );

  // SVG layout. One square per cell; the agent is a ring; the greedy action is
  // a short arrow once the cell has any learned value.
  const cellPx = 64;
  const pad = 6;
  const w = world.width * cellPx;
  const h = world.height * cellPx;
  const centerX = (cell: number) =>
    rowCol(world, cell).col * cellPx + cellPx / 2;
  const centerY = (cell: number) =>
    rowCol(world, cell).row * cellPx + cellPx / 2;
  const inspected = inspect !== null && inspect < cells ? inspect : null;

  const behaviors: { id: Behavior; label: string }[] = [
    { id: "epsilon-greedy", label: "e-greedy" },
    { id: "greedy", label: "greedy" },
    { id: "random", label: "random" },
  ];
  const modes: { id: ClickMode; label: string }[] = [
    { id: "inspect", label: "Inspect" },
    { id: "wall", label: "Toggle wall" },
    { id: "goal", label: "Move goal" },
    { id: "teleport", label: "Drop agent" },
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
          viewBox={`${-pad} ${-pad} ${w + pad * 2} ${h + pad * 2}`}
          className="w-full"
          role="img"
          aria-label={`A ${world.width} by ${world.height} grid. The agent is at cell ${agent}. Each cell is shaded by its learned value and shows an arrow for the best move.`}
        >
          {Array.from({ length: cells }, (_, cell) => {
            const { row, col } = rowCol(world, cell);
            const x = col * cellPx;
            const y = row * cellPx;
            const wall = isWall(world, cell);
            const goal = cell === world.goal;
            const penalty = isPenalty(world, cell);
            const value = values[cell] ?? 0;
            const t = value <= 0 ? 0 : Math.min(1, value / vmax);
            const opacity = wall || goal ? 1 : 0.1 + 0.82 * t;
            const fill = wall ? WALL : penalty ? PENALTY_FILL : ACCENT;
            const visited = ACTIONS.some((a) => (visits[cell]?.[a] ?? 0) > 0);
            const arrowDir = at(ARROW, greedyPolicyAction(q, cell));
            const cx = centerX(cell);
            const cy = centerY(cell);
            return (
              <g
                key={cell}
                role="button"
                tabIndex={0}
                aria-label={`cell ${cell}, value ${value.toFixed(2)}`}
                aria-pressed={cell === inspected}
                className="cursor-pointer outline-none"
                onClick={() => onCellClick(cell)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onCellClick(cell);
                  }
                }}
                onMouseEnter={() => setInspect(cell)}
                onFocus={() => setInspect(cell)}
              >
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={cellPx - 2}
                  height={cellPx - 2}
                  rx={5}
                  fill={fill}
                  fillOpacity={penalty ? 0.4 : opacity}
                  stroke={cell === inspected ? ACCENT : "var(--color-border)"}
                  strokeWidth={cell === inspected ? 2.5 : 1}
                />
                {goal && (
                  <text
                    x={cx}
                    y={cy + 6}
                    textAnchor="middle"
                    fill="#fff"
                    style={{ fontSize: 18, fontWeight: 700 }}
                  >
                    ★
                  </text>
                )}
                {penalty && (
                  <text
                    x={cx}
                    y={cy + 5}
                    textAnchor="middle"
                    className="fill-foreground"
                    style={{ fontSize: 14 }}
                  >
                    !
                  </text>
                )}
                {!wall && !goal && visited && (
                  <line
                    x1={cx - arrowDir.dx * 12}
                    y1={cy - arrowDir.dy * 12}
                    x2={cx + arrowDir.dx * 12}
                    y2={cy + arrowDir.dy * 12}
                    stroke={t > 0.55 ? "#fff" : "var(--color-foreground)"}
                    strokeOpacity={0.85}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    markerEnd="url(#qarrow)"
                  />
                )}
                {!wall && (
                  <text
                    x={x + 5}
                    y={y + 14}
                    className="fill-muted-foreground"
                    fill={t > 0.55 && !goal ? "#fff" : undefined}
                    style={{ fontSize: 9, fontFamily: "var(--font-mono)" }}
                  >
                    {goal ? "" : value.toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}

          <circle
            cx={centerX(agent)}
            cy={centerY(agent)}
            r={11}
            fill="none"
            stroke="var(--color-foreground)"
            strokeWidth={3}
          />
          <circle
            cx={centerX(agent)}
            cy={centerY(agent)}
            r={4}
            fill="var(--color-foreground)"
          />

          <defs>
            <marker
              id="qarrow"
              viewBox="0 0 10 10"
              refX="7"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path
                d="M 0 1 L 8 5 L 0 9"
                fill="none"
                stroke="context-stroke"
                strokeWidth={2}
              />
            </marker>
          </defs>
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground text-xs">Click a cell to:</span>
        {modes.map((mode) => (
          <Toggle
            key={mode.id}
            label={mode.label}
            active={clickMode === mode.id}
            onClick={() => setClickMode(mode.id)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Behaviour policy — how the agent moves while it learns
            </span>
            <div className="flex flex-wrap gap-1.5">
              {behaviors.map((b) => (
                <Toggle
                  key={b.id}
                  label={b.label}
                  active={params.behavior === b.id}
                  onClick={() =>
                    intervene({ type: "setBehavior", value: b.id })
                  }
                />
              ))}
            </div>
          </div>
          <Slider
            label="alpha — how much each new experience overwrites the old estimate"
            value={params.alpha}
            min={0.05}
            max={1}
            step={0.05}
            display={params.alpha.toFixed(2)}
            onChange={(value) => intervene({ type: "setAlpha", value })}
          />
          <Slider
            label="gamma — how much a reward one step away is worth vs. right now"
            value={params.gamma}
            min={0}
            max={0.99}
            step={0.01}
            display={params.gamma.toFixed(2)}
            onChange={(value) => intervene({ type: "setGamma", value })}
          />
          <Slider
            label="epsilon — fraction of moves that are random rather than greedy"
            value={params.epsilon}
            min={0}
            max={1}
            step={0.05}
            display={params.epsilon.toFixed(2)}
            onChange={(value) => intervene({ type: "setEpsilon", value })}
          />
          <Slider
            label="slip — chance a move veers sideways instead of going as intended"
            value={params.slip}
            min={0}
            max={0.5}
            step={0.05}
            display={params.slip.toFixed(2)}
            onChange={(value) => intervene({ type: "setSlip", value })}
          />
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant="secondary"
              onClick={() => intervene({ type: "resetLearning" })}
            >
              Reset Q (keep the world)
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() =>
                intervene({
                  type: "setSeed",
                  value: (Math.floor(Math.random() * 999) + 1),
                })
              }
            >
              New random seed
            </Button>
          </div>
        </SimPanel>

        <SimPanel title="What the agent has learned">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">episode {state.episodes}</Badge>
            <Badge variant="outline">{state.steps} steps</Badge>
            <Badge variant="outline">
              value error {score.valueError.toFixed(3)}
            </Badge>
            <Badge variant="outline">coverage {pct(score.coverage)}</Badge>
            <Badge
              variant={score.greedyOptimal ? "default" : "outline"}
              style={
                score.greedyOptimal
                  ? { backgroundColor: SAGE, color: "#fff" }
                  : undefined
              }
            >
              greedy policy {score.greedyOptimal ? "optimal" : "not yet"}
            </Badge>
          </div>

          {inspected !== null ? (
            <div className="ring-foreground/10 rounded-lg ring-1">
              <div className="text-muted-foreground flex items-center justify-between px-2 py-1 text-xs">
                <span>
                  cell {inspected} ({rowCol(world, inspected).row},
                  {rowCol(world, inspected).col})
                </span>
                <span>V* {(star[inspected] ?? 0).toFixed(2)}</span>
              </div>
              <table className="w-full text-left text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="px-2 py-1 font-medium">move</th>
                    <th className="px-2 py-1 text-right font-medium">Q</th>
                    <th className="px-2 py-1 text-right font-medium">tried</th>
                    <th className="px-2 py-1 text-right font-medium">poke</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {ACTIONS.map((a) => {
                    const best = greedyPolicyAction(q, inspected) === a;
                    return (
                      <tr
                        key={a}
                        className={
                          best ? "text-foreground" : "text-muted-foreground"
                        }
                      >
                        <td className="px-2 py-1">
                          {best ? "▶ " : ""}
                          {ACTION_NAMES[a]}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {(q[inspected]?.[a] ?? 0).toFixed(3)}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {visits[inspected]?.[a] ?? 0}
                        </td>
                        <td className="px-2 py-1 text-right">
                          <button
                            type="button"
                            aria-label={`force ${ACTION_NAMES[a]} value high at cell ${inspected}`}
                            className="text-muted-foreground hover:bg-foreground/10 rounded px-1"
                            onClick={() =>
                              intervene({
                                type: "pokeValue",
                                cell: inspected,
                                action: a,
                                value: (star[inspected] ?? 1) + 1,
                              })
                            }
                          >
                            +
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Hover or click a cell to read its four action-values. Poke one up
              with + to perturb the policy, then watch learning correct it.
            </p>
          )}

          {state.last && (
            <p className="text-muted-foreground text-xs">
              last step took{" "}
              <span className="text-foreground font-mono">
                {ACTION_NAMES[state.last.a]}
              </span>{" "}
              from cell {state.last.x}, paid{" "}
              <span className="text-foreground font-mono">
                {state.last.r.toFixed(2)}
              </span>
              , landed at {state.last.y}, surprise{" "}
              <span className="text-foreground font-mono">
                {state.last.tdError.toFixed(3)}
              </span>
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Finish an episode or flip a control to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The update rule here is exactly the thesis formula: Q(x,a) &larr; (1
        &minus; alpha)Q(x,a) + alpha(r + gamma&middot;max Q(y,&middot;)). The
        agent never sees the world&apos;s map; the value error and the optimal
        badge compare its learned Q against an optimal policy computed
        separately by value iteration, used only for scoring. Each preset runs
        from a fixed random seed so the same preset always produces the same
        run; &ldquo;New random seed&rdquo; restarts learning from a fresh seed.
        This is a small discrete grid; the thesis&apos;s own demo used a
        continuous square, which is the same finite Markov decision process
        idea at higher resolution. The step cost slider was removed from
        controls since its effect duplicates gamma in teaching the discount
        concept; it remains wired in the model at the preset&apos;s value.
      </p>
    </div>
  );
}
