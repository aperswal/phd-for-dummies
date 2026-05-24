"use client";

import { useCallback, useRef } from "react";

import { useAnimationFrame } from "@/components/visualizations/use-animation-frame";

const MAX_STEPS_PER_FRAME = 240;

// Advances a simulation by a fixed virtual interval no matter the device frame
// rate, so behavior is identical at 30fps and 144fps. It accumulates elapsed
// milliseconds and fires `onTick` once per whole interval, capping catch-up so a
// long stall after a background tab can't trigger a runaway burst. The returned
// `reset` zeroes the accumulator, which the caller uses when play toggles or the
// run resets so timing starts clean.
export function useFixedTimestep(
  active: boolean,
  intervalMs: number,
  onTick: () => void,
): () => void {
  const accumulator = useRef(0);

  useAnimationFrame(active, (deltaMs) => {
    accumulator.current += deltaMs;
    let steps = 0;
    while (accumulator.current >= intervalMs && steps < MAX_STEPS_PER_FRAME) {
      accumulator.current -= intervalMs;
      onTick();
      steps += 1;
    }
  });

  return useCallback(() => {
    accumulator.current = 0;
  }, []);
}
