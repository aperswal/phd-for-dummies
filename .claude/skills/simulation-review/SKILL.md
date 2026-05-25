---
name: simulation-review
description: Audit and fix one paper's interactive simulation against a strict UI/UX and correctness rubric. Use when reviewing or repairing a visualization in components/visualizations/<slug>.tsx — wasted space (a small scroll box stranded inside a large reserved box), hidden scroll with no affordance, cramped padding, broken or meaningless color, dials that change nothing or are unlabeled, too many knobs, and logic that doesn't actually respond to the controls. Covers the render context, a control-discipline decision procedure, the full rubric, the per-sim work procedure, the shared-file collision rule, the gates, and a final scoring checklist.
---

# simulation-review

You take one paper's simulation and make it correct, legible, and honest. You audit it against the rubric below, decide what to change, change only that paper's own files, run the gates, and report. You do not redesign the simulation from scratch and you do not touch other papers' files or shared files.

The bar: every control earns its place and visibly does what its label says; no region reserves space it doesn't use; nothing scrolls without telling the reader; every color means one thing and never carries meaning alone; the model honestly responds to every control the reader can touch; and the whole thing reads cleanly at 360px wide and at full width, in light mode and dark mode.

## The render context (memorize this before judging layout)

Every simulation renders inside the paper page:

```
<article class="prose ... mx-auto w-full max-w-3xl px-4 py-10 sm:px-6">
```

So the usable content column is **`max-w-3xl` = 768px**, and there is a left sidebar on `lg` screens. The simulation root is wrapped in `.not-prose`, which **only resets typography, not width**. Consequences you must account for:

- The simulation never gets more than ~768px. A `lg:grid-cols-[1.5fr_1fr]` two-column layout is squeezed into 768px, and `lg:` (1024px viewport) two-column layouts activate only when the viewport is wide while the column stays 768px. Prefer `sm:grid-cols-2` and verify both columns are readable at ~360px each. If a two-column split makes either side unreadable, make it one column.
- On mobile (down to 320–360px) the layout must be single column with **no horizontal scroll** and no element wider than the viewport. Fixed control widths like `w-40`/`w-32` must wrap, not overflow.
- The palette is brand clay-orange `ACCENT #c2683f`, sage green `SAGE` (varies slightly per file, ~`#7d8a6a`), and slate `SLATE`. Dark mode is real and toggleable. Any hardcoded hex (especially `#fff` text on a colored chip, or `bg-*-50` light fills) must stay legible in both themes.

## Control discipline — the dial audit (do this first, it is the heart of the review)

Most of these simulations have too many controls, and some controls do nothing the reader can see. Walk **every** interactive control (every preset button, toggle, slider, text input, click target) and put it in exactly one bucket:

1. **Spine (keep, make prominent).** Preset/scenario buttons that teach the paper's core contrast (e.g. "trust region" vs "no trust region", "ReAct" vs "CoT only"). These are the main thing the reader should touch. They belong at the top.
2. **The paper's lever (keep, label in full).** The one or two knobs that ARE the mechanism the paper is about (the trust-region radius delta, the KL coefficient, the clip epsilon, the exploration constant). Keep these. Make the label a full plain-English statement of what moving it demonstrates, not a cryptic symbol alone. "delta — how far the policy may move each step" beats "δ".
3. **Meta-noise (cut or demote).** Controls that pace or reseed the rendering but are not the paper's idea: `Speed`, `Seed`, and incidental knobs (a discount `gamma` that barely moves the picture, a second noise dial that duplicates the first). These compete with the spine for attention.
4. **Broken (fix the mismatch).** A control rendered as a slider/dial that has only 2–3 meaningful values (make it a toggle/segmented buttons), a control that is shown but changes nothing visible (remove it, or wire it to something real, or convert it to a labeled static readout), or a value that is described as adjustable but is actually static (make the description match reality).

Decision rules for the audit:

- **A control that changes nothing the reader can see does not exist for the reader. Remove it.** A slider you can drag with no visible consequence is worse than no slider.
- **Seed sliders are guilty until proven innocent.** A seed is only worth a control when (a) the reader has a concrete reason to want a different sample and (b) the difference between seeds is legible on screen. Otherwise drop it. If you keep some notion of "try another sample," prefer a single **"Reshuffle / new sample"** button over a 0–20 slider whose individual values mean nothing. If the seed only exists to make the run deterministic, keep it in the model and remove it from the UI; state the determinism in the honesty caption instead.
- **Speed sliders stay only if the run has motion worth pacing.** If a sim advances in a few discrete steps the reader drives by clicking Step, a continuous speed dial is noise. If you keep Speed, label it as playback pace (it controls how fast the animation plays, nothing in the paper).
- **Few, deliberate, fully-labeled.** A control the reader changes once and leaves (a one-shot mode toggle) is fine, but its label must say exactly what it showcases. Never make the reader guess what a knob is for. If you cannot write a one-line plain-English label that says what moving it teaches, the control should not be there.
- When you remove a UI control, also remove the now-dead handler, action type, and state it fed, and delete or update the model test that covered it. Do not leave dead code. Do not delete a test merely to make removal pass — if a test breaks, the behavior it asserted changed, so update the assertion to match the new intended behavior.

## The rubric (score each dimension PASS / FAIL with evidence)

### A. Control discipline
Every control is in bucket 1 or 2, or has been cut/converted/relabeled per the audit. No dial changes nothing. No slider with 2–3 values. Every kept control has a full plain-English label. The spine (presets) is visually dominant; meta-knobs are demoted or gone.

### B. Space usage (no wasted real estate)
No region reserves height it routinely doesn't fill. The named failure: **a small scrollable inner box stranded inside a much larger bordered/reserved box** (e.g. ReAct's `min-h-[20rem]` Trajectory panel wrapping a `max-h-[24rem]` scroller — when the run is short you get a big empty frame; the inner box looks lost in the outer box). Fix by making the container size to its content (drop the fixed `min-h`), or by letting the content use the reserved space, so reserved ≈ used. A panel should look intentional whether it holds 2 rows or 20.

### C. Scroll affordance and autoscroll
Every `overflow-y-auto` / `overflow-auto` region must **tell the reader more content exists**. macOS overlay scrollbars are invisible at rest, so content silently hides — this is the exact bug the reader hits on the ReAct trace. Add a real affordance: a gradient fade mask at the scrollable edge that appears only when there is overflow, or a persistent thin styled scrollbar, or an explicit "+N more" / "scroll for more" hint. And when autoplay appends entries, the container must **follow to the newest item** (scroll to the active step) so the live action is never below the fold. Match scroll direction to list order (newest-first logs scroll to top; trajectories that append at the bottom scroll to bottom).

### D. Padding, sizing, breathing room
No text crammed against a panel edge. Use the established panel padding (`p-4`) — do not tighten it to squeeze content in; resize the content or the layout instead. Respect the type scale already in use (`text-xs` for secondary metadata is the floor; body/answer content should not be shrunk below it to fit). Group related controls with whitespace, not extra borders. Every interactive target stays comfortably tappable (don't shrink buttons below `size="sm"` to fit a row — wrap instead).

### E. Color semantics and dark mode
Within one simulation a color means exactly one thing. The collision to hunt for: `ACCENT` (clay-orange) is both the brand highlight AND the "bad / over-budget / off-target" signal in many sims — so the same color marks "the selected thing" and "the broken thing," which is ambiguous. Pick one meaning per color per sim. Sage = good/healthy/on-target/progress; clay = the problem/the warning; use a neutral (muted/foreground) for plain selection so selection never reads as an error. **Never encode meaning with color alone** (CLAUDE.md): every colored signal also carries a glyph or word (✓ / ✗ / "over budget"). Verify all hardcoded hex (`#fff` on chips, `bg-*-50` fills, `text-*-900`) has adequate contrast in both light and dark mode; prefer the theme tokens (`text-foreground`, `var(--color-*)`, the `dark:` variants already used) over raw hex where a token exists.

### F. Logic correctness, determinism, reset, terminal state
The model must honestly respond to every control the reader can still touch: flip every preset, drag every kept knob to both extremes, toggle every mode, and confirm the readout changes and the change is rule-consistent with the paper (not cosmetic). Specific checks:
- **Reset truly resets**: state, the fixed-timestep clock (`resetClock()`), any inspect/selection, any text-input buffer, and the preset selection. After Reset the sim is byte-identical to first load of that preset.
- **Terminal state**: when the run is solved/done/stuck, autoplay stops (no spinning), Step is disabled or a no-op, and the terminal banner is shown.
- **Indexing under `noUncheckedIndexedAccess`**: no `arr[i]` that can be `undefined` flowing into arithmetic or render; use the `at()` helper / guards already in the models (`lib/simulation/array`).
- **Determinism**: same preset + same inputs ⇒ same run (seeded RNG, fixed timestep). The model test should assert at least one concrete trajectory.
- **Event-log rewind/restore** (where present) restores a faithful snapshot.

### G. Responsiveness
Single column and no horizontal scroll at 360px. SVGs use `viewBox` + `className="w-full"` (no fixed pixel `width` that overflows). Two-column grids degrade to one column and each column is readable at its squeezed width inside 768px. Control rows wrap (`flex-wrap`) instead of overflowing.

### H. Accessibility
Sliders have `aria-label`; SVG scenes have `role="img"` + a descriptive `aria-label` that states the current state; toggles report `aria-pressed`; the transport buttons keep their labels; focus is visible; everything is keyboard-operable. Don't regress what's already there.

### I. Honesty caption
The closing muted paragraph that says what is real vs stubbed stays, stays accurate, and stays short. If you change behavior (e.g. remove a seed, change what a control does), update the caption to match. If you removed determinism controls from the UI, state the determinism here instead.

## Per-sim work procedure

1. **Read everything for this paper**: `components/visualizations/<slug>.tsx`, `<slug>-model.ts`, `<slug>-model.test.ts`, and `content/papers/<slug>/doc.mdx` (so your fixes match what the prose claims the demo does). Read the paper's `meta.json` headline if you need the framing.
2. **Run the control audit** (section above). Write down each control and its bucket and your decision before editing.
3. **Walk the rubric A–I** against the current code. Note every FAIL with a line reference.
4. **Fix**, smallest change that satisfies the rubric. Stay in the file's existing idiom — many sims inline their own `Panel`/`Slider`/`EventLog`; keep that local style, don't swap to shared components mid-file unless the swap IS the fix and is contained to this file. Keep the deterministic model/view split intact (view renders state; model owns rules).
5. **No dead code**: when you cut a control, cut its handler, action variant, and state, and update the test.
6. **Run the gates** (below). Fix anything red. Re-run until green.
7. **Report** in the format below.

## Shared-file rule (critical for parallel runs)

These files are imported by many simulations. **Do not edit them.** If your fix requires changing one, stop and report it as a "shared change needed" item instead — the orchestrator applies shared fixes once so 22 agents don't collide:

- `components/simulation/event-log.tsx` (EventLogPanel — used by 22 sims)
- `components/simulation/sim-panel.tsx` (SimPanel — used by 22 sims)
- `components/simulation/use-fixed-timestep.ts` (used by 24 sims)
- `components/visualizations/play-pause-step-controls.tsx`
- `components/visualizations/visualization-boundary.tsx`
- `components/visualizations/use-animation-frame.ts`
- anything under `lib/simulation/` (`rng`, `history`, `array`, `mdp`)
- `components/ui/*` (shadcn primitives)

You may edit **only**: `components/visualizations/<your-slug>.tsx`, `<your-slug>-model.ts`, and `<your-slug>-model.test.ts`. If a sim inlines its own Panel/log/slider (9 of them do), those live in your `.tsx` and are yours to fix.

## Gates (run scoped, then the build)

```bash
# scoped first — fast feedback on your file
pnpm exec tsc --noEmit
pnpm exec vitest run components/visualizations/<slug>-model.test.ts
pnpm exec eslint components/visualizations/<slug>.tsx components/visualizations/<slug>-model.ts

# then the repo-wide build (the real gate). If many agents run at once,
# the orchestrator runs this once after all agents finish, not per agent.
pnpm build
```

Prettier runs on commit via lint-staged; do not hand-format. Never weaken a type, never `// eslint-disable` to pass, never delete a test to make a change pass.

## Report format (end your run with this)

```
PAPER: <slug>
CONTROLS BEFORE -> AFTER: <e.g. 2 sliders + 3 toggles + seed -> 3 toggles + 1 labeled knob; removed Speed (no motion to pace) and Seed (illegible)>
RUBRIC FAILS FIXED: A:<note> B:<note> C:<note> ... (only the ones that were failing)
SHARED CHANGE NEEDED: <none | file + what + why> 
GATES: tsc <ok/fail> | vitest <ok/fail> | eslint <ok/fail>
NOTES: <anything the orchestrator or a human should know>
```

## Final scoring checklist (every box must be checked or explicitly N/A with reason)

- [ ] A. Every control is spine or paper-lever, or was cut/converted/relabeled. No dial changes nothing. No 2–3-value slider. Every kept control fully labeled. Seed/Speed justified or gone.
- [ ] B. No region reserves space it doesn't fill. No small scroll box stranded in a big empty box.
- [ ] C. Every scroll region shows an affordance when it overflows, and autoplay scrolls to the newest item.
- [ ] D. Padding intact (`p-4`), no cramped text, type scale respected, targets tappable.
- [ ] E. One meaning per color, never color alone, hardcoded hex legible in light and dark.
- [ ] F. Model responds honestly to every remaining control; reset fully resets; terminal stops play; safe indexing; deterministic.
- [ ] G. No horizontal scroll at 360px; SVGs fluid; grids degrade; rows wrap.
- [ ] H. Accessibility preserved (aria-labels, role=img, aria-pressed, keyboard, focus).
- [ ] I. Honesty caption present, accurate, short.
- [ ] Gates: scoped tsc + the sim's vitest + eslint all green (build is the orchestrator's final gate).
- [ ] No dead code left from removed controls; no shared file edited (shared changes flagged, not applied).
