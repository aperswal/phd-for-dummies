"use client";

import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";

interface PlayPauseStepControlsProps {
  playing: boolean;
  onPlayPause: () => void;
  onStep: () => void;
  onReset: () => void;
  disabled?: boolean;
}

// Shared transport for every visualization so play, step, and reset look and
// behave the same across papers. Keyboard-operable buttons with explicit
// labels; the play button reports its state through aria-pressed.
export function PlayPauseStepControls({
  playing,
  onPlayPause,
  onStep,
  onReset,
  disabled = false,
}: PlayPauseStepControlsProps) {
  return (
    <div
      role="group"
      aria-label="Playback controls"
      className="flex flex-wrap items-center gap-2"
    >
      <Button
        type="button"
        size="sm"
        onClick={onPlayPause}
        aria-pressed={playing}
        aria-label={playing ? "Pause" : "Play"}
        disabled={disabled}
      >
        {playing ? (
          <Pause className="size-4" aria-hidden />
        ) : (
          <Play className="size-4" aria-hidden />
        )}
        <span>{playing ? "Pause" : "Play"}</span>
      </Button>
      <Button
        type="button"
        size="sm"
        variant="secondary"
        onClick={onStep}
        aria-label="Step forward one stage"
        disabled={disabled || playing}
      >
        <SkipForward className="size-4" aria-hidden />
        <span>Step</span>
      </Button>
      <Button
        type="button"
        size="sm"
        variant="ghost"
        onClick={onReset}
        aria-label="Reset to the start"
        disabled={disabled}
      >
        <RotateCcw className="size-4" aria-hidden />
        <span>Reset</span>
      </Button>
    </div>
  );
}
