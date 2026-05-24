"use client";

import { useEffect, useRef } from "react";

// Runs `callback` once per frame with the elapsed milliseconds since the last
// frame, but only while `active` is true. Stops and cleans up the moment
// `active` goes false or the component unmounts. The callback is mirrored into a
// ref inside an effect so changing it doesn't restart the loop.
export function useAnimationFrame(
  active: boolean,
  callback: (deltaMs: number) => void,
): void {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  });

  useEffect(() => {
    if (!active) return;
    let frame = 0;
    let last = performance.now();
    const loop = (now: number) => {
      callbackRef.current(now - last);
      last = now;
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [active]);
}
