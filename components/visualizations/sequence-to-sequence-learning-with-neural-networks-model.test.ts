import { describe, expect, it } from "vitest";

import {
  computeSeq,
  EOS,
  getPreset,
  initialState,
  step,
  totalSteps,
  UNK,
  viewAtCursor,
  type SeqParams,
} from "./sequence-to-sequence-learning-with-neural-networks-model";

const LONG: SeqParams = {
  sentenceId: "long",
  reversed: false,
  memory: 4,
  capacity: 12,
  dropEos: false,
  threshold: 0.12,
};

describe("computeSeq invariants", () => {
  it("emits one word per target plus an EOS on the happy path", () => {
    const result = computeSeq({ ...LONG, sentenceId: "short" });
    expect(result.emitted).toHaveLength(result.target.length + 1);
    expect(result.emitted.at(-1)).toBe(EOS);
  });

  it("recovers every word on a short sentence even read forward", () => {
    const result = computeSeq({
      ...LONG,
      sentenceId: "short",
      reversed: false,
    });
    expect(result.allRecovered).toBe(true);
    expect(result.accuracy).toBe(1);
  });

  it("is deterministic: same params give the identical run", () => {
    const a = computeSeq(LONG);
    const b = computeSeq(LONG);
    expect(a.emitted).toEqual(b.emitted);
    expect(a.traces.map((t) => t.signal)).toEqual(
      b.traces.map((t) => t.signal),
    );
  });
});

describe("the reversal trick (the paper's key claim)", () => {
  it("reversing leaves the total lag across all words unchanged", () => {
    const forward = computeSeq({ ...LONG, reversed: false });
    const reversed = computeSeq({ ...LONG, reversed: true });
    const sum = (r: ReturnType<typeof computeSeq>) =>
      r.traces.reduce((acc, t) => acc + t.lag, 0);
    expect(sum(reversed)).toBe(sum(forward));
  });

  it("reversing shrinks the lag on the first word from n to 1", () => {
    const forward = computeSeq({ ...LONG, reversed: false });
    const reversed = computeSeq({ ...LONG, reversed: true });
    const n = forward.source.length;
    expect(forward.traces[0]?.lag).toBe(n);
    expect(reversed.traces[0]?.lag).toBe(1);
  });

  it("the forward long sentence fails on its early words, the reversed one recovers them", () => {
    const forward = computeSeq({ ...LONG, reversed: false });
    const reversed = computeSeq({ ...LONG, reversed: true });
    expect(forward.traces[0]?.emitted).toBe(UNK);
    expect(forward.allRecovered).toBe(false);
    expect(reversed.traces[0]?.emitted).toBe(reversed.target[0]);
    expect(reversed.recoveredCount).toBeGreaterThan(forward.recoveredCount);
  });
});

describe("interventions change the outcome under the model's own rules", () => {
  it("dropping the stop token makes the decoder run away past the sentence length", () => {
    const normal = computeSeq({ ...LONG, sentenceId: "cat", dropEos: false });
    const runaway = computeSeq({ ...LONG, sentenceId: "cat", dropEos: true });
    expect(normal.runaway).toBe(false);
    expect(runaway.runaway).toBe(true);
    expect(runaway.emitted.length).toBeGreaterThan(normal.emitted.length);
    expect(runaway.allRecovered).toBe(false);
  });

  it("more memory recovers more words on the failing forward sentence", () => {
    const weak = computeSeq({ ...LONG, reversed: false, memory: 4 });
    const strong = computeSeq({ ...LONG, reversed: false, memory: 30 });
    expect(strong.recoveredCount).toBeGreaterThan(weak.recoveredCount);
  });

  it("an overfull jar attenuates signal so capacity pressure rises", () => {
    const roomy = computeSeq({ ...LONG, capacity: 12 });
    const tight = computeSeq({ ...LONG, capacity: 3 });
    expect(roomy.capacityPressure).toBe(0);
    expect(tight.capacityPressure).toBeGreaterThan(0);
    expect(tight.recoveredCount).toBeLessThanOrEqual(roomy.recoveredCount);
  });
});

describe("the stepped reducer", () => {
  it("a tick advances the cursor and an intervention resets it to the start", () => {
    let state = initialState("forward-long");
    state = step(state, { kind: "tick" });
    state = step(state, { kind: "tick" });
    expect(state.cursor).toBe(2);
    state = step(state, {
      kind: "intervene",
      action: { type: "toggleReversed" },
    });
    expect(state.cursor).toBe(0);
    expect(state.params.reversed).toBe(true);
    expect(state.log.events.at(-1)?.label).toContain("reversed");
  });

  it("the cursor never runs past the total number of steps", () => {
    let state = initialState("short-clean");
    const total = totalSteps(computeSeq(state.params));
    for (let i = 0; i < total + 10; i++) {
      state = step(state, { kind: "tick" });
    }
    expect(state.cursor).toBe(total);
  });

  it("restoring a logged snapshot replays the exact earlier run", () => {
    let state = initialState("forward-long");
    const original = { ...state.params };
    state = step(state, {
      kind: "intervene",
      action: { type: "toggleReversed" },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "restore", params: original, label: "rewind" },
    });
    expect(state.params).toEqual(original);
    expect(computeSeq(state.params).emitted).toEqual(
      computeSeq(original).emitted,
    );
  });
});

describe("cursor view phases", () => {
  it("walks encode then decode then done", () => {
    const result = computeSeq({ ...getPreset("short-clean").params });
    const n = result.encodeLen;
    expect(viewAtCursor(result, 0).phase).toBe("encode");
    expect(viewAtCursor(result, n).phase).toBe("decode");
    expect(viewAtCursor(result, totalSteps(result)).phase).toBe("done");
  });
});
