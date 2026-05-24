import { describe, expect, it } from "vitest";

import {
  BASE_STYLES,
  getPreset,
  initialState,
  N_STYLES,
  readout,
  softmax,
  step,
  type SimState,
} from "./training-language-models-to-follow-instructions-with-human-feedback-model";

function advance(state: SimState, ticks: number): SimState {
  let current = state;
  for (let i = 0; i < ticks; i++) current = step(current, { kind: "tick" });
  return current;
}

const helpfulIndex = BASE_STYLES.findIndex((style) => style.name === "Helpful");
const sycophantIndex = BASE_STYLES.findIndex(
  (style) => style.name === "Sycophantic",
);

describe("the policy distribution", () => {
  it("stays a probability distribution every tick", () => {
    let state = initialState("aligned");
    for (let i = 0; i < 60; i++) {
      const { pi } = readout(state);
      const total = pi.reduce((sum, value) => sum + value, 0);
      expect(total).toBeCloseTo(1, 6);
      expect(pi.every((value) => value >= 0)).toBe(true);
      state = step(state, { kind: "tick" });
    }
  });

  it("starts at the SFT reference, so initial KL is zero", () => {
    const { kl } = readout(initialState("aligned"));
    expect(kl).toBeCloseTo(0, 6);
  });
});

describe("determinism", () => {
  it("reproduces the same run from the same seed and inputs", () => {
    const a = advance(initialState("hacking"), 40);
    const b = advance(initialState("hacking"), 40);
    expect(readout(b).pi).toEqual(readout(a).pi);
    expect(b.theta).toEqual(a.theta);
  });
});

describe("the aligned preset", () => {
  it("climbs toward Helpful with true quality rising alongside proxy reward", () => {
    const start = readout(initialState("aligned"));
    const end = readout(advance(initialState("aligned"), 80));
    expect(end.proxyReward).toBeGreaterThan(start.proxyReward);
    expect(end.trueQuality).toBeGreaterThan(start.trueQuality);
    const argmax = end.pi.indexOf(Math.max(...end.pi));
    expect(argmax).toBe(helpfulIndex);
  });
});

describe("reward hacking under a corrupted reward model", () => {
  it("drives proxy reward up while true quality peaks and then falls", () => {
    const state = initialState("hacking");
    let peakTrue = 0;
    let current = state;
    for (let i = 0; i < 120; i++) {
      peakTrue = Math.max(peakTrue, readout(current).trueQuality);
      current = step(current, { kind: "tick" });
    }
    const end = readout(current);
    // proxy reward keeps climbing well past where a calibrated model could go.
    expect(end.proxyReward).toBeGreaterThan(1);
    // true quality ends below its own peak: the policy chased the cheat.
    expect(end.trueQuality).toBeLessThan(peakTrue - 0.1);
    // the policy collapses onto the over-rated Sycophantic style.
    const argmax = end.pi.indexOf(Math.max(...end.pi));
    expect(argmax).toBe(sycophantIndex);
  });

  it("does not hack when the reward model is honest (bias removed)", () => {
    let state = initialState("hacking");
    state = step(state, {
      kind: "intervene",
      action: { type: "corruptReward", index: sycophantIndex, amount: -0.55 },
    });
    const end = readout(advance(state, 120));
    const argmax = end.pi.indexOf(Math.max(...end.pi));
    expect(argmax).toBe(helpfulIndex);
  });
});

describe("the KL leash", () => {
  it("keeps the policy closer to SFT as beta grows", () => {
    const weak = readout(
      advance(
        step(initialState("aligned"), {
          kind: "intervene",
          action: { type: "setBeta", value: 0.01 },
        }),
        80,
      ),
    );
    const strong = readout(
      advance(
        step(initialState("aligned"), {
          kind: "intervene",
          action: { type: "setBeta", value: 0.6 },
        }),
        80,
      ),
    );
    expect(strong.kl).toBeLessThan(weak.kl);
  });
});

describe("the alignment tax and the pretraining mix", () => {
  it("recovers capability when gamma (ptx) is turned up, at a small cost to true quality", () => {
    const ppo = readout(advance(initialState("tax"), 120));
    const ptx = readout(
      advance(
        step(initialState("tax"), {
          kind: "intervene",
          action: { type: "setGamma", value: 0.3 },
        }),
        120,
      ),
    );
    expect(ptx.capability).toBeGreaterThan(ppo.capability);
    // the labeler-preference cost of the tax recovery is small.
    expect(ppo.trueQuality - ptx.trueQuality).toBeLessThan(0.1);
  });
});

describe("interventions", () => {
  it("loads a preset by id and resets to the SFT reference", () => {
    const state = step(advance(initialState("aligned"), 30), {
      kind: "intervene",
      action: { type: "loadPreset", id: "tax" },
    });
    expect(state.params).toMatchObject(getPreset("tax").params);
    expect(readout(state).kl).toBeCloseTo(0, 6);
  });

  it("records every intervention in the log with a restorable snapshot", () => {
    const state = step(initialState("aligned"), {
      kind: "intervene",
      action: { type: "setBeta", value: 0.4 },
    });
    const last = state.log.events.at(-1);
    expect(last?.label).toContain("KL beta");
    expect(last?.snapshot.params.beta).toBe(0.4);
  });

  it("restores a snapshot, rewinding the run to that exact moment", () => {
    const mid = advance(initialState("aligned"), 20);
    const snapshot = {
      theta: [...mid.theta],
      thetaSft: [...mid.thetaSft],
      step: mid.step,
      rngState: mid.rngState,
      params: { ...mid.params, rmBias: [...mid.params.rmBias] },
    };
    const moved = advance(mid, 20);
    const restored = step(moved, {
      kind: "intervene",
      action: { type: "restore", snapshot, label: "rewind" },
    });
    expect(restored.theta).toEqual(snapshot.theta);
    expect(restored.step).toBe(snapshot.step);
  });
});

describe("softmax helper", () => {
  it("turns a flat preference vector into a uniform distribution", () => {
    const uniform = softmax(new Array<number>(N_STYLES).fill(0));
    for (const value of uniform) expect(value).toBeCloseTo(1 / N_STYLES, 6);
  });
});
