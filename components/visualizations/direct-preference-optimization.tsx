"use client";

import {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  datasetLoss,
  implicitReward,
  initialState,
  policyKl,
  PRESETS,
  PROMPTS,
  softmax,
  step,
  type Intervention,
  type SimState,
  type Snapshot,
} from "@/components/visualizations/direct-preference-optimization-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#7c8a5a";

type ViewAction =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention }
  | { kind: "reset"; presetId: string };

function reducer(state: SimState, action: ViewAction): SimState {
  if (action.kind === "reset") return initialState(action.presetId);
  return step(state, action);
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

// A fixed-timestep clock layered on the shared frame loop so the gradient steps
// run at the same virtual rate on any device. Returns a reset that zeroes the
// accumulator when play toggles or the run resets.
function useFixedClock(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): () => void {
  const accumulator = useRef(0);
  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < 240) {
      accumulator.current -= intervalMs;
      onTick();
      steps += 1;
    }
  });
  return useCallback(() => {
    accumulator.current = 0;
  }, []);
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1">
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
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

interface PromptViewProps {
  state: SimState;
  promptId: string;
  inspect: { promptId: string; index: number } | null;
  onInspect: (promptId: string, index: number) => void;
}

// One prompt's panel: a row of bars per completion. The filled bar is the
// policy's current probability; a thin sage tick marks where the reference
// started. A small clay-orange dot above a bar marks the dataset's preferred
// completion, an open ring marks the dispreferred one.
function PromptView({ state, promptId, inspect, onInspect }: PromptViewProps) {
  const prompt = PROMPTS.find((item) => item.id === promptId);
  if (!prompt) return null;

  const probs = softmax(state.logits[promptId] ?? []);
  const refProbs = softmax(state.refLogits[promptId] ?? []);
  const pair = state.dataset.find((item) => item.promptId === promptId);
  const winner = pair ? (pair.flipped ? pair.loser : pair.winner) : -1;
  const loser = pair ? (pair.flipped ? pair.winner : pair.loser) : -1;

  const width = 480;
  const rowH = 46;
  const labelW = 150;
  const barX = labelW + 8;
  const barW = width - barX - 64;
  const top = 26;

  return (
    <div className="ring-foreground/10 overflow-hidden rounded-xl ring-1">
      <svg
        viewBox={`0 0 ${width} ${top + prompt.completions.length * rowH + 8}`}
        className="w-full"
        role="img"
        aria-label={`Policy probabilities for the prompt '${prompt.text}'. Preferred completion '${prompt.completions[winner]?.text ?? "none"}', dispreferred '${prompt.completions[loser]?.text ?? "none"}'.`}
      >
        <text
          x={12}
          y={16}
          className="fill-muted-foreground"
          style={{ fontSize: 12 }}
        >
          prompt: {prompt.text}
        </text>
        {prompt.completions.map((completion, index) => {
          const y = top + index * rowH;
          const prob = probs[index] ?? 0;
          const ref = refProbs[index] ?? 0;
          const reward = implicitReward(state, promptId, index);
          const isWinner = index === winner;
          const isLoser = index === loser;
          const isInspect =
            inspect?.promptId === promptId && inspect.index === index;
          return (
            <g
              key={`${promptId}-${index}`}
              role="button"
              tabIndex={0}
              aria-label={`Inspect completion ${completion.text}`}
              className="cursor-pointer outline-none"
              onClick={() => onInspect(promptId, index)}
              onFocus={() => onInspect(promptId, index)}
              onMouseEnter={() => onInspect(promptId, index)}
            >
              <rect
                x={2}
                y={y - 2}
                width={width - 4}
                height={rowH - 6}
                rx={6}
                fill={isInspect ? "var(--color-muted)" : "transparent"}
              />
              <text
                x={12}
                y={y + 17}
                className="fill-foreground"
                style={{ fontSize: 12.5, fontWeight: isWinner ? 600 : 400 }}
              >
                {completion.text}
              </text>
              {isWinner && (
                <circle cx={labelW - 6} cy={y + 13} r={4} fill={ACCENT} />
              )}
              {isLoser && (
                <circle
                  cx={labelW - 6}
                  cy={y + 13}
                  r={4}
                  fill="none"
                  stroke={ACCENT}
                  strokeWidth={1.5}
                />
              )}
              <rect
                x={barX}
                y={y}
                width={barW}
                height={20}
                rx={4}
                className="fill-foreground/5"
              />
              <rect
                x={barX}
                y={y}
                width={Math.max(2, barW * prob)}
                height={20}
                rx={4}
                fill={
                  isWinner
                    ? ACCENT
                    : isLoser
                      ? ACCENT_SOFT
                      : "var(--color-muted-foreground)"
                }
                fillOpacity={isWinner || isLoser ? 1 : 0.55}
              />
              <line
                x1={barX + barW * ref}
                y1={y - 2}
                x2={barX + barW * ref}
                y2={y + 22}
                stroke={SAGE}
                strokeWidth={2}
              />
              <text
                x={barX + barW + 6}
                y={y + 14}
                className="fill-muted-foreground"
                style={{
                  fontSize: 11,
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                {percent(prob)}
              </text>
              <text
                x={barX + barW + 6}
                y={y + 14}
                dx={36}
                className="fill-muted-foreground"
                style={{
                  fontSize: 10,
                  fontFamily: "var(--font-mono, monospace)",
                }}
              >
                {reward >= 0 ? "+" : ""}
                {reward.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function DirectPreferenceOptimization() {
  const [state, dispatch] = useReducer(reducer, undefined, () =>
    initialState("align"),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("align");
  const [inspect, setInspect] = useState<{
    promptId: string;
    index: number;
  } | null>(null);

  const { params, dataset } = state;
  const loss = datasetLoss(state);
  const kl = policyKl(state);

  const intervene = useCallback(
    (action: Intervention) => dispatch({ kind: "intervene", action }),
    [],
  );

  const resetClock = useFixedClock(playing, 520 / speed, () =>
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

  const onInspect = useCallback((promptId: string, index: number) => {
    setInspect({ promptId, index });
  }, []);

  const onRestore = useCallback(
    (snapshot: Snapshot, label: string) => {
      setPlaying(false);
      resetClock();
      dispatch({
        kind: "intervene",
        action: { type: "restore", snapshot, label: `rewind: ${label}` },
      });
    },
    [resetClock],
  );

  const inspected = inspect
    ? PROMPTS.find((p) => p.id === inspect.promptId)?.completions[inspect.index]
    : null;

  // Scroll the event log to the top whenever a new entry is added. The log
  // displays newest-first (reversed), so the active entry is always at the top.
  const logRef = useRef<HTMLDivElement>(null);
  const logLength = state.log.events.length;
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0;
  }, [logLength]);

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
        {PROMPTS.map((prompt) => (
          <PromptView
            key={prompt.id}
            state={state}
            promptId={prompt.id}
            inspect={inspect}
            onInspect={onInspect}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs">
        <Badge variant="secondary">step {state.tick}</Badge>
        <Badge variant="outline">DPO loss {loss.toFixed(3)}</Badge>
        <Badge variant="outline">KL(policy || ref) {kl.toFixed(3)}</Badge>
        <Badge variant="outline">
          sigma weight {params.useAdaptiveWeight ? "on" : "off"}
        </Badge>
        <span className="text-muted-foreground flex items-center gap-1.5">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{ background: ACCENT }}
          />
          preferred
          <span
            className="ml-2 inline-block size-2.5 rounded-full"
            style={{ boxShadow: `0 0 0 1.5px ${ACCENT}` }}
          />
          dispreferred
          <span
            className="ml-2 inline-block h-3 w-0.5"
            style={{ background: SAGE }}
          />
          reference start
        </span>
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
            label="Playback speed (animation pace only)"
            value={speed}
            min={0.5}
            max={4}
            step={0.5}
            display={`${speed.toFixed(1)}x`}
            onChange={setSpeed}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Controls">
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Adaptive sigma weight"
              active={params.useAdaptiveWeight}
              onClick={() => intervene({ type: "toggleAdaptiveWeight" })}
            />
          </div>
          <Slider
            label="beta — how tightly the KL constraint holds the policy near the reference"
            value={params.beta}
            min={0.05}
            max={2}
            step={0.05}
            display={params.beta.toFixed(2)}
            onChange={(value) => intervene({ type: "setBeta", value })}
          />
          <Slider
            label="Step size — gradient update magnitude per tick"
            value={params.learningRate}
            min={0.2}
            max={3}
            step={0.1}
            display={params.learningRate.toFixed(1)}
            onChange={(value) => intervene({ type: "setLearningRate", value })}
          />
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-xs">
              Preference pairs (click to mislabel)
            </span>
            {dataset.map((pair, index) => {
              const prompt = PROMPTS.find((p) => p.id === pair.promptId);
              const w = pair.flipped ? pair.loser : pair.winner;
              const l = pair.flipped ? pair.winner : pair.loser;
              return (
                <div
                  key={`${pair.promptId}-${index}`}
                  className="flex items-center gap-1.5"
                >
                  <button
                    type="button"
                    onClick={() => intervene({ type: "flipPair", index })}
                    className="ring-foreground/10 hover:bg-foreground/5 flex-1 rounded-md px-2 py-1 text-left text-xs ring-1"
                    aria-pressed={pair.flipped}
                  >
                    <span className="font-medium">
                      {prompt?.completions[w]?.text}
                    </span>{" "}
                    &gt;{" "}
                    <span className="text-muted-foreground">
                      {prompt?.completions[l]?.text}
                    </span>
                    {pair.flipped && (
                      <span
                        className="ml-1 text-[10px]"
                        style={{ color: ACCENT }}
                      >
                        mislabeled
                      </span>
                    )}
                  </button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    aria-label={`Remove pair ${index + 1}`}
                    onClick={() => intervene({ type: "removePair", index })}
                  >
                    remove
                  </Button>
                </div>
              );
            })}
            {dataset.length === 0 && (
              <p className="text-muted-foreground text-xs">
                No pairs left, so the gradient is zero. Reset to bring them
                back.
              </p>
            )}
          </div>
        </Panel>

        <Panel title="Inside the implicit reward">
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="secondary">
              {inspected ? `'${inspected.text}'` : "hover a completion"}
            </Badge>
            {inspect && (
              <Badge variant="outline">
                reward{" "}
                {implicitReward(state, inspect.promptId, inspect.index).toFixed(
                  3,
                )}
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground text-xs">
            The reward is beta times the log of how much likelier the policy
            makes this reply than the reference does. No reward model was
            trained; the policy is the reward model. The gradient leans on the
            pair by{" "}
            {params.useAdaptiveWeight
              ? "beta times sigma(reward of the loser minus reward of the winner), so a pair already ranked right barely moves."
              : "a constant beta with the sigma weight off, so it keeps shoving the winner up even after it has won."}
          </p>
          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            {/* Fade mask at the bottom signals hidden content when scrollable. */}
            <div className="relative">
              <div
                ref={logRef}
                className="ring-foreground/10 max-h-32 overflow-y-auto rounded-lg ring-1"
              >
                {state.log.events.length === 0 ? (
                  <p className="text-muted-foreground px-2 py-1.5 text-xs">
                    Step, drag a slider, or flip a pair to start the log.
                  </p>
                ) : (
                  <ul className="text-xs">
                    {[...state.log.events].reverse().map((event) => (
                      <li key={event.id}>
                        <button
                          type="button"
                          className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                          onClick={() => onRestore(event.snapshot, event.label)}
                        >
                          {event.id}. {event.label}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {state.log.events.length > 3 && (
                <div
                  className="from-background pointer-events-none absolute bottom-0 left-0 right-0 h-6 rounded-b-lg bg-gradient-to-t to-transparent"
                  aria-hidden="true"
                />
              )}
            </div>
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs the exact DPO loss and its gradient over {PROMPTS.length}{" "}
        prompts with {PROMPTS[0]?.completions.length ?? 4} candidate replies
        each, where a real run works over a full vocabulary of completions. The
        policy starts as a copy of the reference, and the reward is beta times
        log pi(y)/pi_ref(y), exactly the paper&apos;s implicit reward. The
        reference starts at a uniform distribution (seed fixed for
        determinism); every run of the same preset produces the same result.
      </p>
    </div>
  );
}
