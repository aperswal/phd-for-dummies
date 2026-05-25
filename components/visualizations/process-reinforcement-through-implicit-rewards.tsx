"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  initialState,
  MOVE_LABELS,
  MOVES,
  policyDistribution,
  PRESETS,
  step,
  type HistoryPoint,
  type Intervention,
  type Move,
  type PrimeState,
} from "@/components/visualizations/process-reinforcement-through-implicit-rewards-model";

const ACCENT = "#c2683f"; // clay-orange: reward hacking signal / true accuracy line
const SAGE = "#6f7d5f"; // sage: good/healthy — PRM accuracy line, correct move bars
const SLATE = "#7c8a99"; // slate: reported reward line, shortcut move bars

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: PrimeState, action: ViewAction): PrimeState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

// Sage marks the good/correct move; clay-orange is reserved for the hacking
// warning so the same color never means both "correct" and "problem" on screen.
const MOVE_COLOR: Record<Move, string> = {
  correct: SAGE,
  shortcut: SLATE,
  wrong: "var(--color-muted-foreground)",
};

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

const CHART_W = 1000;
const CHART_H = 260;
const PAD_L = 44;
const PAD_R = 18;
const PAD_T = 16;
const PAD_B = 30;

function linePath(
  history: HistoryPoint[],
  pick: (point: HistoryPoint) => number,
  maxTick: number,
): string {
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;
  const span = Math.max(maxTick, 1);
  return history
    .map((point, i) => {
      const x = PAD_L + (point.tick / span) * innerW;
      const y = PAD_T + (1 - pick(point)) * innerH;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

export function ImplicitRewardTraining() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("online"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.5);
  const [presetId, setPresetId] = useState("online");

  const { params, history, lastBatch, log } = state;
  const policy = policyDistribution(state);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedTimestep(playing, 420 / speed, () =>
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

  // Scroll the batch table to the bottom whenever a new batch arrives so the
  // live action is always visible rather than silently hidden below the fold.
  const batchTableRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = batchTableRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [lastBatch]);

  const maxTick = Math.max(state.tick, 30);
  const innerH = CHART_H - PAD_T - PAD_B;
  const hacking = state.hackGap > 0.18;

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

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-3.5 rounded-sm"
            style={{ background: ACCENT }}
          />
          true accuracy
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-3.5 rounded-sm"
            style={{ background: SLATE }}
          />
          reported reward
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-3.5 rounded-sm"
            style={{ background: SAGE }}
          />
          PRM accuracy
        </span>
      </div>

      <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
        <svg
          viewBox={`0 0 ${CHART_W} ${CHART_H}`}
          className="w-full"
          role="img"
          aria-label={`Training curves over ${state.tick} steps. True accuracy ${pct(
            state.trueAccuracy,
          )}, reported reward ${pct(state.reportedReward)}, PRM accuracy ${pct(
            state.prmAccuracy,
          )}.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((g) => {
            const y = PAD_T + (1 - g) * innerH;
            return (
              <g key={`grid-${g}`}>
                <line
                  x1={PAD_L}
                  y1={y}
                  x2={CHART_W - PAD_R}
                  y2={y}
                  className="stroke-foreground/10"
                  strokeWidth={1}
                />
                <text
                  x={PAD_L - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-muted-foreground"
                  style={{ fontSize: 10 }}
                >
                  {pct(g)}
                </text>
              </g>
            );
          })}

          <path
            d={linePath(history, (p) => p.reportedReward, maxTick)}
            fill="none"
            stroke={SLATE}
            strokeWidth={2}
            strokeOpacity={0.85}
          />
          <path
            d={linePath(history, (p) => p.prmAccuracy, maxTick)}
            fill="none"
            stroke={SAGE}
            strokeWidth={2}
            strokeDasharray="5 4"
            strokeOpacity={0.9}
          />
          <path
            d={linePath(history, (p) => p.trueAccuracy, maxTick)}
            fill="none"
            stroke={ACCENT}
            strokeWidth={2.6}
          />

          <text
            x={CHART_W - PAD_R}
            y={CHART_H - 8}
            textAnchor="end"
            className="fill-muted-foreground"
            style={{ fontSize: 11 }}
          >
            training step
          </text>
        </svg>
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
            label="Playback speed (not a paper parameter)"
            value={speed}
            min={0.5}
            max={4}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div
        className="flex flex-wrap items-center gap-3 rounded-xl px-4 py-3 ring-1"
        style={{
          borderColor: "transparent",
          background: hacking ? "rgba(194,104,63,0.08)" : "var(--color-muted)",
        }}
        aria-live="polite"
      >
        <span
          className="inline-block size-3 rounded-full"
          style={{ background: hacking ? ACCENT : SAGE }}
          aria-hidden
        />
        <span className="text-sm font-medium">
          {hacking ? "Reward hacking" : "Reward tracks truth"}
        </span>
        <span className="text-muted-foreground text-xs">
          reward minus true accuracy {state.hackGap >= 0 ? "+" : ""}
          {state.hackGap.toFixed(2)}
        </span>
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className="ml-auto"
          onClick={() => intervene({ type: "injectShortcut" })}
        >
          Inject shortcut
        </Button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              PRM update mode
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label={
                  params.prmOnline
                    ? "Online (keeps reward honest)"
                    : "Frozen (reward drifts)"
                }
                active={params.prmOnline}
                onClick={() => intervene({ type: "togglePrmOnline" })}
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              PRM initialization
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label={
                  params.prmFromScratch
                    ? "From scratch (zero reward)"
                    : "From SFT model"
                }
                active={!params.prmFromScratch}
                onClick={() => intervene({ type: "togglePrmFromScratch" })}
              />
            </div>
          </div>
          <Slider
            label="beta — how much weight the implicit process reward carries (Eq 3)"
            value={params.beta}
            min={0.1}
            max={1.5}
            step={0.05}
            display={params.beta.toFixed(2)}
            onChange={(value) => intervene({ type: "setBeta", value })}
          />
          <Slider
            label="K — rollouts per prompt; more rollouts stabilize the leave-one-out advantage"
            value={params.rollouts}
            min={2}
            max={16}
            step={1}
            display={String(params.rollouts)}
            onChange={(value) => intervene({ type: "setRollouts", value })}
          />
          <Slider
            label="Policy learning rate — how aggressively the policy updates each step"
            value={params.policyLr}
            min={0.1}
            max={1.2}
            step={0.05}
            display={params.policyLr.toFixed(2)}
            onChange={(value) => intervene({ type: "setPolicyLr", value })}
          />
          <Slider
            label="PRM learning rate — how fast the online PRM corrects its preferences"
            value={params.prmLr}
            min={0}
            max={1}
            step={0.05}
            display={params.prmLr.toFixed(2)}
            onChange={(value) => intervene({ type: "setPrmLr", value })}
          />
        </SimPanel>

        <SimPanel title="Inside the loop">
          <div className="flex flex-col gap-2">
            <span className="text-muted-foreground text-xs font-medium">
              Policy over the branch
            </span>
            {MOVES.map((move) => (
              <div
                key={`pol-${move}`}
                className="flex items-center gap-2 text-xs"
              >
                <span className="text-muted-foreground w-28 shrink-0">
                  {MOVE_LABELS[move]}
                </span>
                <span className="bg-foreground/5 h-2.5 flex-1 overflow-hidden rounded-sm">
                  <span
                    className="block h-full rounded-sm"
                    style={{
                      width: `${policy[move] * 100}%`,
                      background: MOVE_COLOR[move],
                    }}
                  />
                </span>
                <span className="w-10 text-right font-mono">
                  {pct(policy[move])}
                </span>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">step {state.tick}</Badge>
            <Badge variant="outline">true acc {pct(state.trueAccuracy)}</Badge>
            <Badge variant="outline">reward {pct(state.reportedReward)}</Badge>
            <Badge variant="outline">PRM acc {pct(state.prmAccuracy)}</Badge>
          </div>

          {/* Wrapper provides the scroll affordance: the fade mask at the bottom
              signals more content is below, since macOS overlay scrollbars
              are invisible at rest. */}
          <div className="relative">
            <div
              ref={batchTableRef}
              className="ring-foreground/10 max-h-40 overflow-y-auto rounded-lg ring-1"
            >
              <table className="w-full text-left text-xs">
                <thead className="bg-card text-muted-foreground sticky top-0">
                  <tr>
                    <th className="px-2 py-1 font-medium">rollout</th>
                    <th className="px-2 py-1 text-right font-medium">
                      verifier
                    </th>
                    <th className="px-2 py-1 text-right font-medium">
                      process r
                    </th>
                    <th className="px-2 py-1 text-right font-medium">
                      advantage
                    </th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {lastBatch.length === 0 ? (
                    <tr>
                      <td
                        className="text-muted-foreground px-2 py-1.5"
                        colSpan={4}
                      >
                        Step once to sample a batch of rollouts.
                      </td>
                    </tr>
                  ) : (
                    lastBatch.map((row, i) => (
                      <tr
                        key={`batch-${i}`}
                        className={
                          row.outcome > 0
                            ? "text-foreground"
                            : "text-muted-foreground"
                        }
                      >
                        <td className="px-2 py-1">{MOVE_LABELS[row.move]}</td>
                        <td className="px-2 py-1 text-right">{row.outcome}</td>
                        <td className="px-2 py-1 text-right">
                          {row.processReward.toFixed(2)}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {row.advantage >= 0 ? "+" : ""}
                          {row.advantage.toFixed(2)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            {/* Gradient fade signals more rows are below the visible area */}
            <div
              className="pointer-events-none absolute inset-x-0 bottom-0 h-6 rounded-b-lg"
              style={{
                background:
                  "linear-gradient(to bottom, transparent, var(--color-card, var(--color-background)))",
              }}
              aria-hidden
            />
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step, flip a control, or inject the shortcut to start the log."
              onRestore={(event) =>
                intervene({
                  type: "restore",
                  snapshot: event.snapshot,
                  label: `rewind to step ${event.snapshot.tick}`,
                })
              }
            />
          </div>
        </SimPanel>
      </div>

      <p className="text-muted-foreground text-xs">
        The implicit reward here is exactly equation 3, beta times log of the
        policy model&apos;s token probability over the frozen reference&apos;s.
        The branch has three moves where the paper&apos;s model has thousands of
        tokens, and the outcome verifier is a clean rule where the paper&apos;s
        is exact-match on math and unit tests on code. The online cross-entropy
        update on rollouts is Algorithm 1&apos;s line 8, the one thing that
        keeps the reward honest. Every run is fully deterministic: the same
        preset always produces the same trajectory. Reset restores the preset
        from scratch.
      </p>
    </div>
  );
}
