"use client";

import { useCallback, useReducer, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  GAMES,
  getGame,
  initialState,
  PRESETS,
  recommendationCorrect,
  recommendedMove,
  rootChildren,
  rootValue,
  step,
  trueBestMove,
  type BackupOperator,
  type Intervention,
  type SelectivityRule,
  type SimSnapshot,
  type SimState,
  type TreeNode,
} from "@/components/visualizations/efficient-selectivity-and-backup-operators-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#7d8a6a";
// Destructive red for pruned moves — distinct from ACCENT so "pruned" and "plays" never share a color.
const PRUNE_COLOR = "#b91c1c";

const MOVE_FILLS = ["#c2683f", "#7d8a6a", "#a8763f", "#5d6b78"];

const BACKUPS: { value: BackupOperator; label: string; description: string }[] =
  [
    {
      value: "mean",
      label: "Mean",
      description: "Plain average of all rollouts — underestimates at high counts",
    },
    {
      value: "max",
      label: "Max",
      description:
        "Best child value — overestimates because lucky outliers dominate",
    },
    {
      value: "robust-max",
      label: "Robust max",
      description: "Value of the most-visited child — more stable than raw max",
    },
    {
      value: "mix",
      label: "Mix (paper's best)",
      description:
        "Slides from mean toward robust max as visits grow — the paper's recommended operator",
    },
  ];

const SELECTIVITIES: { value: SelectivityRule; label: string; description: string }[] =
  [
    {
      value: "coulom",
      label: "Coulom urgency",
      description:
        "Picks by probability-of-being-best; keeps a floor so nothing is abandoned",
    },
    {
      value: "greedy",
      label: "Greedy",
      description: "Always picks the highest-valued move — pours simulations into one option",
    },
    {
      value: "uniform",
      label: "Uniform",
      description: "Picks at random — ignores all value information",
    },
    {
      value: "pruning",
      label: "Progressive pruning",
      description:
        "Cuts off low-scoring moves permanently — can kill the best move after bad luck",
    },
  ];

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function fmt(value: number): string {
  return value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
}

function SimPanel({
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

export function CrazyStoneSearch() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("coulom-mix"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [presetId, setPresetId] = useState("coulom-mix");
  const [inspect, setInspect] = useState<number | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // A fixed-timestep accumulator on top of the shared frame loop, kept in a ref
  // so changing the rate doesn't restart the loop. Each whole interval runs one
  // simulation.
  const accumulator = useRef(0);
  const intervalMs = 1000 / Math.max(speed, 1);
  const MAX_SIMULATIONS = 800;
  useAnimationFrame(playing, (deltaMs) => {
    // Stop the loop once the tree has gathered plenty of simulations so it
    // doesn't spin forever after the answer has settled.
    if (state.simulations >= MAX_SIMULATIONS) {
      setPlaying(false);
      return;
    }
    accumulator.current += deltaMs;
    let runs = 0;
    while (accumulator.current >= intervalMs && runs < 64) {
      accumulator.current -= intervalMs;
      dispatch({ kind: "tick" });
      runs += 1;
    }
    // Autoscroll the event log to the newest entry (rendered reversed, so scroll to top).
    if (logRef.current) logRef.current.scrollTop = 0;
  });

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const onPlayPause = useCallback(() => {
    accumulator.current = 0;
    setPlaying((value) => !value);
  }, []);
  const onStep = useCallback(() => dispatch({ kind: "tick" }), []);
  const onReset = useCallback(() => {
    setPlaying(false);
    accumulator.current = 0;
    setInspect(null);
    dispatch({ kind: "reset", presetId });
  }, [presetId]);

  const loadPreset = useCallback((id: string) => {
    setPlaying(false);
    accumulator.current = 0;
    setPresetId(id);
    setInspect(null);
    dispatch({ kind: "intervene", action: { type: "loadPreset", id } });
  }, []);

  const restore = useCallback(
    (snapshot: SimSnapshot, label: string) =>
      dispatch({
        kind: "intervene",
        action: { type: "restore", snapshot, label },
      }),
    [],
  );

  const game = getGame(state.params.gameId);
  const children = rootChildren(state);
  const rec = recommendedMove(state);
  const trueBest = trueBestMove(state);
  const correct = recommendationCorrect(state);
  const root = rootValue(state);

  const totalVisits = children.reduce((sum, c) => sum + c.visits, 0) || 1;
  const maxVisits = children.reduce((m, c) => Math.max(m, c.visits), 1);

  // SVG layout: the root sits at the top, each move is a column with a bar for
  // visit share and a value readout. The most recently walked path lights up.
  const width = 1000;
  const height = 360;
  const rootX = width / 2;
  const rootY = 44;
  const colCount = children.length;
  const colW = (width - 80) / colCount;
  const moveY = 150;
  const barTop = 200;
  const barMax = 110;
  const colX = (i: number) => 40 + (i + 0.5) * colW;

  const lastPathSet = new Set(state.lastPath);

  const inspectNode: TreeNode | null =
    inspect !== null
      ? (children.find((c) => c.moveIndex === inspect) ?? null)
      : null;

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
          aria-label={`Search tree after ${state.simulations} simulations. The search recommends move ${rec === null ? "none yet" : (children.find((c) => c.moveIndex === rec)?.label ?? "?")}, and the true best move is ${game.moves[trueBest]?.label ?? "?"}.`}
        >
          {children.map((child, i) => {
            const onPath = lastPathSet.has(child.id);
            return (
              <line
                key={`edge-${child.id}`}
                x1={rootX}
                y1={rootY + 18}
                x2={colX(i)}
                y2={moveY - 22}
                stroke={onPath ? ACCENT : "var(--color-border)"}
                strokeWidth={onPath ? 3 : 1 + (child.visits / totalVisits) * 6}
                strokeOpacity={onPath ? 0.9 : 0.4}
              />
            );
          })}

          <g>
            <circle
              cx={rootX}
              cy={rootY}
              r={18}
              fill="var(--color-muted)"
              stroke={ACCENT}
              strokeWidth={2}
            />
            <text
              x={rootX}
              y={rootY + 4}
              textAnchor="middle"
              className="fill-foreground"
              style={{ fontSize: 12, fontWeight: 600 }}
            >
              {fmt(root)}
            </text>
            <text
              x={rootX}
              y={rootY - 26}
              textAnchor="middle"
              className="fill-muted-foreground"
              style={{ fontSize: 11 }}
            >
              root value
            </text>
          </g>

          {children.map((child, i) => {
            const isRec = child.moveIndex === rec;
            const isTrue = child.moveIndex === trueBest;
            const isPruned = state.pruned.includes(child.moveIndex);
            const fill = MOVE_FILLS[i % MOVE_FILLS.length] ?? ACCENT;
            const x = colX(i);
            const onPath = lastPathSet.has(child.id);
            return (
              <g
                key={`move-${child.id}`}
                role="button"
                tabIndex={0}
                aria-label={`Inspect move ${child.label}`}
                aria-pressed={inspect === child.moveIndex}
                className="cursor-pointer outline-none"
                onClick={() => setInspect(child.moveIndex)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setInspect(child.moveIndex);
                  }
                }}
                onMouseEnter={() => setInspect(child.moveIndex)}
              >
                <circle
                  cx={x}
                  cy={moveY}
                  r={20}
                  fill={isPruned ? "var(--color-muted)" : fill}
                  fillOpacity={isPruned ? 0.3 : onPath ? 1 : 0.85}
                  stroke={isRec ? ACCENT : isTrue ? SAGE : "transparent"}
                  strokeWidth={isRec || isTrue ? 3 : 0}
                  strokeDasharray={isTrue && !isRec ? "4 3" : undefined}
                />
                <text
                  x={x}
                  y={moveY + 5}
                  textAnchor="middle"
                  className="fill-white"
                  fillOpacity={isPruned ? 0.5 : 1}
                  style={{ fontSize: 14, fontWeight: 700 }}
                >
                  {child.label}
                </text>
                {isPruned && (
                  <text
                    x={x}
                    y={moveY - 28}
                    textAnchor="middle"
                    fill={PRUNE_COLOR}
                    style={{ fontSize: 10 }}
                  >
                    ✕ pruned
                  </text>
                )}
                {isRec && !isPruned && (
                  <text
                    x={x}
                    y={moveY - 28}
                    textAnchor="middle"
                    fill={ACCENT}
                    style={{ fontSize: 10, fontWeight: 600 }}
                  >
                    ▶ plays
                  </text>
                )}

                <rect
                  x={x - colW * 0.32}
                  y={barTop}
                  width={colW * 0.64}
                  height={barMax}
                  rx={3}
                  className="fill-foreground/5"
                />
                <rect
                  x={x - colW * 0.32}
                  y={barTop + barMax - (child.visits / maxVisits) * barMax}
                  width={colW * 0.64}
                  height={(child.visits / maxVisits) * barMax}
                  rx={3}
                  fill={isPruned ? ACCENT_SOFT : fill}
                  fillOpacity={isPruned ? 0.4 : 0.85}
                />
                <text
                  x={x}
                  y={barTop + barMax + 16}
                  textAnchor="middle"
                  className="fill-muted-foreground"
                  style={{ fontSize: 11 }}
                >
                  {child.visits} sims
                </text>
                <text
                  x={x}
                  y={barTop + barMax + 30}
                  textAnchor="middle"
                  className="fill-foreground"
                  style={{ fontSize: 11, fontWeight: 600 }}
                >
                  {fmt(child.value)}
                </text>
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
        />
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={() => intervene({ type: "burst", count: 50 })}
            disabled={playing}
          >
            +50 sims
          </Button>
          <div className="w-36">
            <Slider
              label="Playback speed (sims/s)"
              value={speed}
              min={1}
              max={30}
              step={1}
              display={`${speed}`}
              onChange={setSpeed}
            />
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs font-medium">
              Backup operator — how a node scores itself from its children
            </span>
            <div className="flex flex-wrap gap-1.5">
              {BACKUPS.map((item) => (
                <Toggle
                  key={item.value}
                  label={item.label}
                  active={state.params.backup === item.value}
                  onClick={() =>
                    intervene({ type: "setBackup", value: item.value })
                  }
                />
              ))}
            </div>
            <span className="text-muted-foreground text-xs">
              {BACKUPS.find((b) => b.value === state.params.backup)?.description}
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs font-medium">
              Selectivity — which move gets the next simulation
            </span>
            <div className="flex flex-wrap gap-1.5">
              {SELECTIVITIES.map((item) => (
                <Toggle
                  key={item.value}
                  label={item.label}
                  active={state.params.selectivity === item.value}
                  onClick={() =>
                    intervene({ type: "setSelectivity", value: item.value })
                  }
                />
              ))}
            </div>
            <span className="text-muted-foreground text-xs">
              {SELECTIVITIES.find((s) => s.value === state.params.selectivity)?.description}
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs font-medium">
              Tree scenario — which set of moves the search explores
            </span>
            <div className="flex flex-wrap gap-1.5">
              {GAMES.map((item) => (
                <Toggle
                  key={item.id}
                  label={item.label}
                  active={state.params.gameId === item.id}
                  onClick={() => intervene({ type: "setGame", id: item.id })}
                />
              ))}
            </div>
          </div>
          <Slider
            label="Expand threshold — simulations before a node becomes internal and uses the backup operator"
            value={state.params.expandThreshold}
            min={1}
            max={16}
            step={1}
            display={String(state.params.expandThreshold)}
            onChange={(value) => intervene({ type: "setThreshold", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the search">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{state.simulations} simulations</Badge>
            <Badge variant="outline">root {fmt(root)}</Badge>
            <Badge
              variant="outline"
              className={
                correct === false
                  ? "border-destructive text-destructive"
                  : correct
                    ? "border-foreground/40"
                    : undefined
              }
            >
              {correct === null
                ? "no recommendation yet"
                : correct
                  ? "recommends the best move"
                  : "recommends a worse move"}
            </Badge>
          </div>

          <div className="ring-foreground/10 rounded-lg p-3 text-xs ring-1">
            {inspectNode ? (
              <div className="flex flex-col gap-1">
                <span className="text-foreground font-medium">
                  Move {inspectNode.label}
                </span>
                <span className="text-muted-foreground">
                  visits S = {inspectNode.visits}, backed-up value ={" "}
                  {fmt(inspectNode.value)}
                </span>
                <span className="text-muted-foreground">
                  share of simulations ={" "}
                  {Math.round((inspectNode.visits / totalVisits) * 100)}%,
                  variance = {inspectNode.variance.toFixed(2)}
                </span>
                <span className="text-muted-foreground">
                  hidden true value = {fmt(inspectNode.trueValue)}, rollout
                  noise = {inspectNode.noise.toFixed(2)}
                  {game.moves[inspectNode.moveIndex]?.trap
                    ? ", the hidden gem"
                    : ""}
                </span>
              </div>
            ) : (
              <span className="text-muted-foreground">
                Hover or click a move to read its statistics.
              </span>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log — click an entry to rewind
            </span>
            {/* Outer wrapper positions the scroll-fade mask without clipping the ring. */}
            <div className="ring-foreground/10 relative rounded-lg ring-1">
              <div
                ref={logRef}
                className="max-h-28 overflow-y-auto rounded-lg"
              >
                {state.log.length === 0 ? (
                  <p className="text-muted-foreground px-2 py-1.5 text-xs">
                    Flip a control or load a preset to start the log, then click
                    an entry to rewind.
                  </p>
                ) : (
                  <ul className="text-xs">
                    {[...state.log].reverse().map((event) => (
                      <li key={event.id}>
                        <button
                          type="button"
                          className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                          onClick={() =>
                            restore(
                              event.snapshot,
                              `rewind to ${event.simulations} sims`,
                            )
                          }
                        >
                          {event.id}. {event.label} ({event.simulations} sims)
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {/* Fade mask signals more content below when the list overflows. */}
              {state.log.length > 3 && (
                <div
                  className="pointer-events-none absolute bottom-0 left-0 right-0 h-6 rounded-b-lg"
                  style={{
                    background:
                      "linear-gradient(to bottom, transparent, var(--color-background))",
                  }}
                />
              )}
            </div>
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        Each tick runs one full simulation: descend by the selectivity rule,
        expand a node, run a rollout, and back the value up. The rollout value
        of a move is sampled around a hidden true value with that move&apos;s
        noise, so a good move with wide noise can look bad at first. The dashed
        sage ring marks the move with the highest true value; the solid clay{" "}
        <span aria-label="triangle right">▶</span> ring marks the move the
        search would play now; a red ✕ marks a pruned move. Each preset uses a
        fixed seed so the run is deterministic and reproducible. This runs a{" "}
        {colCount}-move tree a few plies deep, where Crazy Stone ran thousands
        of moves on a 9 by 9 board.
      </p>
    </div>
  );
}
