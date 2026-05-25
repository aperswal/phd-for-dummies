"use client";

import { useCallback, useMemo, useReducer, useState } from "react";

import { EventLogPanel } from "@/components/simulation/event-log";
import { SimPanel } from "@/components/simulation/sim-panel";
import { useFixedTimestep } from "@/components/simulation/use-fixed-timestep";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import {
  getItem,
  highestTier,
  initialState,
  isTerminal,
  PRESETS,
  step,
  TECH_TREE,
  type Coder,
  type Curriculum,
  type Intervention,
  type Phase,
  type SimState,
} from "@/components/visualizations/voyager-an-open-ended-embodied-agent-model";

const ACCENT = "#c2683f";
const SAGE = "#6f7d5f";
const SLATE = "#5b6670";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

const PHASE_LABEL: Record<Phase, string> = {
  propose: "curriculum proposes a task",
  retrieve: "retrieve skills from library",
  code: "GPT-4 writes the program",
  execute: "run it, read the feedback",
  verify: "self-verification critic",
  commit: "commit skill to library",
  idle: "waiting for the next task",
};

const PHASE_ORDER: Phase[] = [
  "propose",
  "retrieve",
  "code",
  "execute",
  "verify",
  "commit",
];

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
        aria-label={label}
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

const MAX_TIER = TECH_TREE.reduce((max, item) => Math.max(max, item.tier), 0);

export function VoyagerLifelongLoop() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("full"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);
  const [presetId, setPresetId] = useState("full");
  const [inspect, setInspect] = useState<string | null>(null);

  const { params, library, unlocked, active, last, log } = state;
  const unlockedSet = useMemo(() => new Set(unlocked), [unlocked]);
  const librarySet = useMemo(
    () => new Set(library.map((skill) => skill.itemId)),
    [library],
  );
  const terminal = useMemo(() => isTerminal(state), [state]);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  // Stop autoplay when the frontier is exhausted so the animation doesn't spin
  // forever on a settled state. The check is fresh on every render because the
  // callback is not wrapped in useCallback.
  const resetClock = useFixedTimestep(playing, 650 / speed, () => {
    if (isTerminal(state)) {
      setPlaying(false);
      return;
    }
    dispatch({ kind: "tick" });
  });

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);
  const onStep = useCallback(() => {
    if (!terminal) dispatch({ kind: "tick" });
  }, [terminal]);
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

  // Lay the tech tree out as columns by tier so the climb reads left to right.
  const tiers = useMemo(() => {
    const byTier: { tier: number; items: typeof TECH_TREE }[] = [];
    for (let t = 0; t <= MAX_TIER; t++) {
      byTier.push({
        tier: t,
        items: TECH_TREE.filter((item) => item.tier === t),
      });
    }
    return byTier;
  }, []);

  const colWidth = 132;
  const rowHeight = 34;
  const treeWidth = (MAX_TIER + 1) * colWidth;
  const maxRows = Math.max(...tiers.map((column) => column.items.length));
  const treeHeight = maxRows * rowHeight + 20;

  const inspectedSkill =
    inspect !== null
      ? library.find((skill) => skill.itemId === inspect)
      : undefined;
  const inspectedItem = inspect !== null ? getItem(inspect) : null;

  const curricula: { id: Curriculum; label: string }[] = [
    { id: "automatic", label: "automatic" },
    { id: "manual", label: "manual" },
    { id: "random", label: "random" },
  ];
  const coders: { id: Coder; label: string }[] = [
    { id: "gpt-4", label: "GPT-4" },
    { id: "gpt-3.5", label: "GPT-3.5" },
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

      <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
        <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
          <svg
            viewBox={`0 0 ${treeWidth} ${treeHeight + 56}`}
            className="w-full"
            role="img"
            aria-label={`A Minecraft tech tree across ${MAX_TIER + 1} tiers. ${unlocked.length} of ${TECH_TREE.length} items are unlocked. The agent is currently in the ${active ? PHASE_LABEL[state.phase] : "idle"} phase.`}
          >
            {tiers.map((column) =>
              column.items.map((item, rowIndex) => {
                const x = column.tier * colWidth + 6;
                const y = rowIndex * rowHeight + 6;
                const isUnlocked = unlockedSet.has(item.id);
                const inLibrary = librarySet.has(item.id);
                const isActive = active?.itemId === item.id;
                const fill = isActive
                  ? ACCENT
                  : isUnlocked
                    ? inLibrary
                      ? SAGE
                      : SLATE
                    : "var(--color-muted)";
                // ACCENT, SAGE, and SLATE are all mid-dark brand colors;
                // white text on them meets contrast in both light and dark mode.
                const textFill = isActive || isUnlocked ? "#fff" : undefined;
                return (
                  <g
                    key={item.id}
                    role="button"
                    tabIndex={0}
                    aria-label={`${item.label}, ${isUnlocked ? "unlocked" : "locked"}${inLibrary ? ", in skill library" : ""}. Click to force this task.`}
                    aria-pressed={inspect === item.id}
                    className="cursor-pointer outline-none"
                    onClick={() => {
                      setInspect(item.id);
                      intervene({ type: "forceTask", itemId: item.id });
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setInspect(item.id);
                        intervene({ type: "forceTask", itemId: item.id });
                      }
                    }}
                    onMouseEnter={() => setInspect(item.id)}
                    onFocus={() => setInspect(item.id)}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={colWidth - 14}
                      height={rowHeight - 8}
                      rx={6}
                      fill={fill}
                      fillOpacity={isUnlocked || isActive ? 1 : 0.4}
                      stroke={
                        inspect === item.id
                          ? ACCENT
                          : isActive
                            ? "#fff"
                            : "var(--color-border)"
                      }
                      strokeWidth={inspect === item.id || isActive ? 2 : 1}
                      strokeDasharray={isActive ? "4 3" : undefined}
                    />
                    <text
                      x={x + 9}
                      y={y + (rowHeight - 8) / 2 + 4}
                      fill={textFill}
                      className={textFill ? undefined : "fill-foreground"}
                      style={{ fontSize: 10.5, fontWeight: 500 }}
                    >
                      {item.label}
                    </text>
                  </g>
                );
              }),
            )}

            {tiers.map((column) => (
              <text
                key={`tier-${column.tier}`}
                x={column.tier * colWidth + 6}
                y={treeHeight + 14}
                className="fill-muted-foreground"
                style={{ fontSize: 10 }}
              >
                tier {column.tier}
              </text>
            ))}

            {PHASE_ORDER.map((phase, index) => {
              const isCurrent = state.phase === phase && active !== null;
              const x = index * (treeWidth / PHASE_ORDER.length) + 6;
              return (
                <g key={phase}>
                  <circle
                    cx={x + 7}
                    cy={treeHeight + 38}
                    r={6}
                    fill={isCurrent ? ACCENT : "var(--color-muted)"}
                    stroke={isCurrent ? ACCENT : "var(--color-border)"}
                  />
                  <text
                    x={x + 18}
                    y={treeHeight + 42}
                    className={
                      isCurrent ? "fill-foreground" : "fill-muted-foreground"
                    }
                    style={{ fontSize: 9.5, fontWeight: isCurrent ? 600 : 400 }}
                  >
                    {phase}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <SimPanel title="Skill library" className="min-w-52 lg:w-56">
          {library.length === 0 ? (
            <p className="text-muted-foreground text-xs">
              {params.skillLibrary
                ? "Empty. Verified skills land here and become callable for harder tasks."
                : "Off. Every task is coded from scratch, so nothing accumulates."}
            </p>
          ) : (
            <div className="relative">
              <ul className="flex max-h-72 flex-col gap-1 overflow-y-auto text-xs">
                {library.map((skill) => (
                  <li
                    key={skill.itemId}
                    className="ring-foreground/10 flex items-center justify-between gap-2 rounded-md px-2 py-1 ring-1"
                    style={{
                      borderLeft: `3px solid ${skill.composedFrom > 0 ? SAGE : SLATE}`,
                    }}
                  >
                    <span className="truncate">{skill.label}</span>
                    {skill.composedFrom > 0 && (
                      <span className="text-muted-foreground shrink-0 font-mono text-[10px]">
                        +{skill.composedFrom}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
              {/* Gradient fade at the bottom tells the reader more items scroll below. */}
              {library.length > 8 && (
                <div
                  className="pointer-events-none absolute right-0 bottom-0 left-0 h-8 rounded-b-sm"
                  style={{
                    background:
                      "linear-gradient(to bottom, transparent, var(--color-background))",
                  }}
                />
              )}
            </div>
          )}
          <p className="text-muted-foreground text-[10px]">
            A green bar marks a skill composed from earlier skills. The number
            is how many it reused.
          </p>
        </SimPanel>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="min-w-28 w-36 flex-shrink-0">
          <Slider
            label="Playback speed (animation pace only, does not affect the paper)"
            value={speed}
            min={1}
            max={8}
            step={1}
            display={`${speed}×`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <SimPanel title="Controls">
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Curriculum — how the agent picks its next task
            </span>
            <div className="flex flex-wrap gap-1.5">
              {curricula.map((option) => (
                <Toggle
                  key={option.id}
                  label={option.label}
                  active={params.curriculum === option.id}
                  onClick={() =>
                    intervene({ type: "setCurriculum", value: option.id })
                  }
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Coder — which LLM writes the programs
            </span>
            <div className="flex flex-wrap gap-1.5">
              {coders.map((option) => (
                <Toggle
                  key={option.id}
                  label={option.label}
                  active={params.coder === option.id}
                  onClick={() =>
                    intervene({ type: "setCoder", value: option.id })
                  }
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs">
              Components — toggle Voyager&apos;s key ablations
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label="Skill library"
                active={params.skillLibrary}
                onClick={() => intervene({ type: "toggleSkillLibrary" })}
              />
              <Toggle
                label="Self-verification"
                active={params.selfVerification}
                onClick={() => intervene({ type: "toggleSelfVerification" })}
              />
            </div>
          </div>
          <Slider
            label="Hallucination rate — how often the curriculum proposes non-existent items"
            value={params.hallucinationRate}
            min={0}
            max={0.6}
            step={0.05}
            display={`${Math.round(params.hallucinationRate * 100)}%`}
            onChange={(value) => intervene({ type: "setHallucination", value })}
          />
        </SimPanel>

        <SimPanel title="What the agent has done">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">{unlocked.length} items</Badge>
            <Badge variant="outline">{state.iterations} iterations</Badge>
            <Badge variant="outline">tier {highestTier(state)}</Badge>
            <Badge variant="outline">{state.failedAttempts} shelved</Badge>
            {/* destructive variant stays legible in both light and dark mode */}
            <Badge
              variant={state.committedUnverified > 0 ? "destructive" : "outline"}
            >
              {state.committedUnverified > 0 ? "⚠ " : ""}
              {state.committedUnverified} unverified
            </Badge>
          </div>

          <div className="ring-foreground/10 rounded-lg px-2 py-1.5 text-xs ring-1">
            <span className="text-muted-foreground">phase</span>{" "}
            <span className="text-foreground font-medium">
              {PHASE_LABEL[state.phase]}
            </span>
            {active && (
              <span className="text-muted-foreground">
                {" "}
                on <span className="text-foreground">{active.label}</span>,
                round {active.round}
              </span>
            )}
          </div>

          {inspectedItem ? (
            <div className="ring-foreground/10 rounded-lg px-2 py-1.5 text-xs ring-1">
              <div className="flex items-center justify-between">
                <span className="text-foreground font-medium">
                  {inspectedItem.label}
                </span>
                <span className="text-muted-foreground">
                  tier {inspectedItem.tier}
                </span>
              </div>
              <p className="text-muted-foreground mt-1">
                {inspectedItem.needs.length > 0
                  ? `composes from ${inspectedItem.needs.map((id) => getItem(id).label).join(", ")}`
                  : "no prerequisites"}
              </p>
              {inspectedSkill ? (
                <p className="text-muted-foreground">
                  in the library
                  {inspectedSkill.composedFrom > 0
                    ? `, built by reusing ${inspectedSkill.composedFrom} earlier skill${inspectedSkill.composedFrom === 1 ? "" : "s"}`
                    : ""}
                </p>
              ) : (
                <p className="text-muted-foreground">not learned yet</p>
              )}
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              Hover or click any item to read its prerequisites. Clicking forces
              the curriculum to attempt that task next.
            </p>
          )}

          {last && (
            <p className="text-muted-foreground text-xs">
              last: <span className="text-foreground">{last.detail}</span>
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <EventLogPanel
              events={log.events}
              emptyHint="Step the loop or flip a control to start the log."
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

      {terminal && (
        <div className="ring-foreground/10 rounded-lg px-3 py-2 ring-1">
          <p className="text-muted-foreground text-xs">
            <span className="text-foreground font-medium">Run complete.</span>{" "}
            The frontier is exhausted — no further reachable tasks remain.
            Reset or load a different preset to run again.
          </p>
        </div>
      )}

      <p className="text-muted-foreground text-xs">
        The loop is the paper&apos;s real control flow: the curriculum proposes
        a task, relevant skills are retrieved, the coder writes a program with
        up to four rounds of refinement from environment feedback and execution
        errors, a self-verification critic gates the commit, and a verified
        program joins the skill library keyed by its description. The world is a
        small {TECH_TREE.length}-item tech tree and GPT-4&apos;s coding is
        modelled as a per-round success chance that rises with retrieved skills;
        the real agent plays Minecraft through Mineflayer and queries GPT-4 over
        thousands of items. Each preset uses a fixed seed so the same preset
        always produces the same run; playback speed is animation pace only.
      </p>
    </div>
  );
}
