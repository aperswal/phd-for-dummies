"use client";

import { useCallback, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PlayPauseStepControls } from "@/components/visualizations/play-pause-step-controls";
import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";
import {
  ANSWER_KINDS,
  initialState,
  policyProbs,
  PRESETS,
  referenceProbs,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/deepseekmath-pushing-the-limits-model";

const ACCENT = "#c2683f";
const ACCENT_SOFT = "#e0b59f";
const SAGE = "#7c8a5a";
const INK = "#2b2622";

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function signed(value: number): string {
  const fixed = value.toFixed(2);
  return value > 0 ? `+${fixed}` : fixed;
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
  step: stepSize,
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
        step={stepSize}
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

function Panel({
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

// A small line chart of the emergent quantities over training steps. Maj@K and
// Pass@K share a frame so the reader sees Maj@K climb while Pass@K stays flat.
function HistoryChart({ state }: { state: SimState }) {
  const points = state.history;
  const width = 480;
  const height = 150;
  const pad = 26;
  const last = points[points.length - 1];
  const maxStep = Math.max(20, last?.step ?? 20);

  const x = (s: number) => pad + (s / maxStep) * (width - pad * 2);
  const y = (v: number) => height - pad - v * (height - pad * 2);

  const line = (key: "majAtK" | "passAtK" | "pCorrect") =>
    points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.step)} ${y(p[key])}`)
      .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`Maj@K is ${percent(last?.majAtK ?? 0)} and Pass@K is ${percent(
        last?.passAtK ?? 0,
      )} after ${last?.step ?? 0} steps.`}
    >
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line
            x1={pad}
            x2={width - pad}
            y1={y(v)}
            y2={y(v)}
            stroke={INK}
            strokeOpacity={0.12}
          />
          <text
            x={4}
            y={y(v) + 3}
            fill={INK}
            fillOpacity={0.5}
            style={{ fontSize: 9 }}
          >
            {percent(v)}
          </text>
        </g>
      ))}
      <path
        d={line("pCorrect")}
        fill="none"
        stroke={ACCENT_SOFT}
        strokeWidth={1.5}
      />
      <path d={line("passAtK")} fill="none" stroke={SAGE} strokeWidth={2} />
      <path d={line("majAtK")} fill="none" stroke={ACCENT} strokeWidth={2.5} />
      <text
        x={pad}
        y={height - 6}
        fill={INK}
        fillOpacity={0.5}
        style={{ fontSize: 9 }}
      >
        training steps
      </text>
    </svg>
  );
}

export function GroupRelativePolicyOptimization() {
  const [state, setState] = useState<SimState>(() => initialState("sharpen"));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [presetId, setPresetId] = useState("sharpen");

  const accumulator = useRef(0);
  const intervalMs = 420 / speed;

  useAnimationFrame(playing, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < 30) {
      accumulator.current -= intervalMs;
      setState((current) => step(current, { kind: "tick" }));
      steps += 1;
    }
  });

  const resetClock = useCallback(() => {
    accumulator.current = 0;
  }, []);

  const intervene = useCallback((action: Intervention) => {
    setState((current) => step(current, { kind: "intervene", action }));
  }, []);

  const onPlayPause = useCallback(() => {
    resetClock();
    setPlaying((value) => !value);
  }, [resetClock]);

  const onStep = useCallback(() => {
    setState((current) => step(current, { kind: "tick" }));
  }, []);

  const onReset = useCallback(() => {
    setPlaying(false);
    resetClock();
    setState(initialState(presetId));
  }, [presetId, resetClock]);

  const loadPreset = useCallback(
    (id: string) => {
      setPlaying(false);
      resetClock();
      setPresetId(id);
      setState(initialState(id));
    },
    [resetClock],
  );

  const probs = policyProbs(state);
  const refProbs = referenceProbs(state);
  const { params, lastGroup, lastStats } = state;
  const argmax = probs.indexOf(Math.max(...probs));
  const last = state.history[state.history.length - 1];
  const preset = PRESETS.find((item) => item.id === presetId);

  // Group the sampled answers by kind so the reader can see how many of each the
  // group drew and the spread of advantages that produced.
  const groupByKind = ANSWER_KINDS.map((_, kindIndex) =>
    lastGroup.filter((sample) => sample.kindIndex === kindIndex),
  );

  // Layout for the policy bars.
  const width = 520;
  const barAreaTop = 16;
  const barAreaH = 150;
  const slot = (width - 24) / ANSWER_KINDS.length;

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

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="ring-foreground/10 overflow-hidden rounded-xl p-3 ring-1">
          <span className="text-muted-foreground mb-1 block text-xs font-medium">
            Policy over answer kinds
          </span>
          <svg
            viewBox={`0 0 ${width} 210`}
            className="w-full"
            role="img"
            aria-label={`The policy now puts ${percent(
              probs[argmax] ?? 0,
            )} of its mass on '${ANSWER_KINDS[argmax]?.label}'.`}
          >
            {ANSWER_KINDS.map((kind, i) => {
              const x = 12 + i * slot + 4;
              const w = slot - 8;
              const h = (probs[i] ?? 0) * barAreaH;
              const refH = (refProbs[i] ?? 0) * barAreaH;
              const fill = kind.correct ? ACCENT : ACCENT_SOFT;
              return (
                <g key={kind.id}>
                  <rect
                    x={x}
                    y={barAreaTop}
                    width={w}
                    height={barAreaH}
                    rx={3}
                    className="fill-foreground/5"
                  />
                  <rect
                    x={x}
                    y={barAreaTop + (barAreaH - h)}
                    width={w}
                    height={h}
                    rx={3}
                    fill={fill}
                    stroke={i === argmax ? INK : "transparent"}
                    strokeWidth={1.5}
                  />
                  <line
                    x1={x}
                    x2={x + w}
                    y1={barAreaTop + (barAreaH - refH)}
                    y2={barAreaTop + (barAreaH - refH)}
                    stroke={INK}
                    strokeWidth={1.5}
                    strokeDasharray="3 2"
                    strokeOpacity={0.6}
                  />
                  <text
                    x={x + w / 2}
                    y={barAreaTop + (barAreaH - h) - 4}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 10 }}
                  >
                    {percent(probs[i] ?? 0)}
                  </text>
                  <text
                    x={x + w / 2}
                    y={barAreaTop + barAreaH + 14}
                    textAnchor="middle"
                    className="fill-foreground"
                    style={{ fontSize: 9 }}
                  >
                    {kind.correct ? "correct" : "wrong"}
                  </text>
                  <text
                    x={x + w / 2}
                    y={barAreaTop + barAreaH + 26}
                    textAnchor="middle"
                    className="fill-muted-foreground"
                    style={{ fontSize: 8 }}
                  >
                    {kind.label.split(",")[0]}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="text-muted-foreground mt-1 text-[10px]">
            Solid bars are the current policy; the dashed line is the frozen
            reference the KL term pulls toward. Correct kinds are clay, wrong
            kinds are pale.
          </p>
        </div>

        <div className="ring-foreground/10 overflow-hidden rounded-xl p-3 ring-1">
          <span className="text-muted-foreground mb-1 block text-xs font-medium">
            Maj@K and Pass@K over training
          </span>
          <HistoryChart state={state} />
          <div className="text-muted-foreground mt-1 flex flex-wrap gap-3 text-[10px]">
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-3 rounded-sm"
                style={{ background: ACCENT }}
              />
              Maj@K
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-3 rounded-sm"
                style={{ background: SAGE }}
              />
              Pass@K
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-3 rounded-sm"
                style={{ background: ACCENT_SOFT }}
              />
              correct mass
            </span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <PlayPauseStepControls
          playing={playing}
          onPlayPause={onPlayPause}
          onStep={onStep}
          onReset={onReset}
        />
        <div className="flex flex-wrap gap-2 text-xs">
          <Badge variant="secondary">step {state.step}</Badge>
          <Badge variant="outline">
            group mean {lastStats.groupMean.toFixed(2)}
          </Badge>
          <Badge variant="outline">
            group std {lastStats.groupStd.toFixed(2)}
          </Badge>
          <Badge variant="outline">KL {lastStats.klToRef.toFixed(3)}</Badge>
          <Badge variant="outline">Maj@K {percent(last?.majAtK ?? 0)}</Badge>
          <Badge variant="outline">Pass@K {percent(last?.passAtK ?? 0)}</Badge>
        </div>
        <div className="w-36">
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

      <div className="grid gap-4 sm:grid-cols-2">
        <Panel title="Knobs">
          <Slider
            label="Group size G"
            value={params.groupSize}
            min={2}
            max={64}
            step={1}
            display={String(Math.round(params.groupSize))}
            onChange={(value) => intervene({ type: "setGroupSize", value })}
          />
          <Slider
            label="KL coefficient (pull to reference)"
            value={params.klCoef}
            min={0}
            max={0.4}
            step={0.01}
            display={params.klCoef.toFixed(2)}
            onChange={(value) => intervene({ type: "setKlCoef", value })}
          />
          <Slider
            label="Clip epsilon"
            value={params.clipEps}
            min={0.05}
            max={0.6}
            step={0.05}
            display={params.clipEps.toFixed(2)}
            onChange={(value) => intervene({ type: "setClipEps", value })}
          />
          <Slider
            label="Step size"
            value={params.learningRate}
            min={0.1}
            max={1.5}
            step={0.05}
            display={params.learningRate.toFixed(2)}
            onChange={(value) => intervene({ type: "setLearningRate", value })}
          />
          <Slider
            label="Reward model noise"
            value={params.rewardNoise}
            min={0}
            max={1}
            step={0.05}
            display={params.rewardNoise.toFixed(2)}
            onChange={(value) => intervene({ type: "setRewardNoise", value })}
          />
          <div className="flex flex-wrap gap-1.5">
            <Toggle
              label="Divide by group std"
              active={params.normalizeStd}
              onClick={() => intervene({ type: "toggleNormalizeStd" })}
            />
          </div>
          <Slider
            label="Seed"
            value={state.seed}
            min={1}
            max={20}
            step={1}
            display={String(state.seed)}
            onChange={(value) => intervene({ type: "setSeed", value })}
          />
        </Panel>

        <Panel title="Reward model and the sampled group">
          <div className="flex flex-col gap-1.5">
            <span className="text-muted-foreground text-[11px]">
              Make the reward model over-reward one kind, then step and watch
              the advantages follow the score, not the truth.
            </span>
            <div className="flex flex-wrap gap-1.5">
              <Toggle
                label="honest"
                active={params.hackKind < 0}
                onClick={() => intervene({ type: "setHack", kindIndex: -1 })}
              />
              {ANSWER_KINDS.map((kind, index) => (
                <Toggle
                  key={kind.id}
                  label={kind.label.split(",")[0] ?? kind.id}
                  active={params.hackKind === index}
                  onClick={() =>
                    intervene({ type: "setHack", kindIndex: index })
                  }
                />
              ))}
            </div>
            {params.hackKind >= 0 && (
              <Slider
                label="Hack bonus"
                value={params.hackBonus}
                min={0}
                max={3}
                step={0.1}
                display={`+${params.hackBonus.toFixed(1)}`}
                onChange={(value) => intervene({ type: "setHackBonus", value })}
              />
            )}
          </div>

          <div className="ring-foreground/10 max-h-44 overflow-y-auto rounded-lg ring-1">
            <table className="w-full text-left text-xs">
              <thead className="bg-card text-muted-foreground sticky top-0">
                <tr>
                  <th className="px-2 py-1 font-medium">kind</th>
                  <th className="px-2 py-1 text-right font-medium">drawn</th>
                  <th className="px-2 py-1 text-right font-medium">reward</th>
                  <th className="px-2 py-1 text-right font-medium">
                    advantage
                  </th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {groupByKind.map((samples, i) => {
                  if (samples.length === 0) return null;
                  const avgReward =
                    samples.reduce((sum, s) => sum + s.reward, 0) /
                    samples.length;
                  const avgAdv =
                    samples.reduce((sum, s) => sum + s.advantage, 0) /
                    samples.length;
                  const kind = ANSWER_KINDS[i];
                  return (
                    <tr
                      key={kind?.id ?? i}
                      className={
                        kind?.correct
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }
                    >
                      <td className="px-2 py-1">
                        {kind?.label.split(",")[0] ?? "?"}
                      </td>
                      <td className="px-2 py-1 text-right">{samples.length}</td>
                      <td className="px-2 py-1 text-right">
                        {avgReward.toFixed(2)}
                      </td>
                      <td
                        className="px-2 py-1 text-right"
                        style={{ color: avgAdv >= 0 ? ACCENT : SAGE }}
                      >
                        {signed(avgAdv)}
                      </td>
                    </tr>
                  );
                })}
                {lastGroup.length === 0 && (
                  <tr className="text-muted-foreground">
                    <td className="px-2 py-2" colSpan={4}>
                      Step or play to sample a group.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-muted-foreground text-xs font-medium">
              Event log
            </span>
            <div className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1">
              {state.log.length === 0 ? (
                <p className="text-muted-foreground p-2 text-[11px]">
                  Step, play, or move a knob to start the log.
                </p>
              ) : (
                <ul className="divide-foreground/5 divide-y text-[11px]">
                  {[...state.log]
                    .slice(-12)
                    .reverse()
                    .map((entry) => (
                      <li
                        key={entry.id}
                        className="flex items-center justify-between gap-2 px-2 py-1"
                      >
                        <span className="text-muted-foreground truncate">
                          <span className="font-mono">#{entry.step}</span>{" "}
                          {entry.label}
                        </span>
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="h-5 px-1.5 text-[10px]"
                          onClick={() => {
                            setPlaying(false);
                            resetClock();
                            intervene({
                              type: "restore",
                              snapshot: entry.snapshot,
                              label: `rewind to step ${entry.step}`,
                            });
                          }}
                        >
                          rewind
                        </Button>
                      </li>
                    ))}
                </ul>
              )}
            </div>
          </div>
        </Panel>
      </div>

      <p className="text-muted-foreground text-xs">
        This runs the GRPO objective on a softmax policy over{" "}
        {ANSWER_KINDS.length} answer kinds for one question, where the paper
        runs it on a 7B transformer sampling 64 outputs per question. The
        advantage is exactly the paper&apos;s group-relative normalization, the
        reward minus the group mean over the group standard deviation. The
        reward model here is a fixed table you can poison; a trained reward
        model learns its scores from data.
      </p>
    </div>
  );
}
