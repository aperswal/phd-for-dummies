---
name: paper-simulation
description: Design and build a genuinely interactive, perturbable simulation of a paper's core mechanism — a live model the reader pokes, not a slideshow. Use from add-paper (or directly) whenever a paper's idea is a dynamical system, protocol, algorithm, scheduler, or multi-agent process where the interesting thing is what happens when you change inputs, inject faults, or intervene mid-run. Covers the think-first enumeration, the model/view split, deterministic seeded time-stepping, discrete-event and physical engines, the intervention catalog (kill a node, drop a message, partition the network, perturb an entity), observability (inspect entities, event log, scrub and rewind), the controls, and the tests. The bar is that the reader can cause a different, rule-consistent outcome by acting.
---

# paper-simulation

You build a simulation, not a presentation. A presentation is a scripted sequence that plays the same way every time and the reader watches. A simulation is a model with real state and real rules. The screen renders the state, time advances the model by its rules, and the reader can change parameters and inject events at any moment, after which the model keeps going under the new reality and produces an outcome you did not script.

The test, stated plainly: if the reader can never change what happens by acting, you built a presentation and you are not done. Determinism is fine and wanted (same seed, same inputs, same run). What is not fine is the absence of agency. The reader must be able to reach in, do something the paper warns about or you are curious about, and watch the consequence play out honestly.

Two reference experiences, decoded, because they are the whole point:

- **Cells that bounce around and flip white to red once they learn each other's health.** This is information propagation you can watch spread across a population, with a physical layer for legibility (the bouncing) and a logical layer that is the actual content (who knows what about whom). The reader wants to watch the truth diffuse, and to perturb it (kill a node, partition the group) and watch the diffusion change.
- **Killing a node in the gap between `verify` and `accept` and seeing what happens.** This is intervening at a precise point in a protocol's phase sequence, for a specific actor, mid-run, and then watching the real consequence under the protocol's own rules. The scheduled `accept` never arrives, a timeout fires, quorum logic reacts, and the reader learns whether the protocol stalls, recovers, or violates safety.

Both share the DNA you are building toward: a model you can hit with an unplanned, well-timed event and a view that shows you the inside, not just the outcome.

## Step 1. Think through the whole space before writing any code

This is the part the reader most wants you to do. Do not jump to a component. First enumerate everything a curious person would want to poke at, on paper, in your reasoning. Run the **clear-thinking-writing** lenses here. Answer all of these explicitly before building:

- **Entities and their state.** What are the actors (nodes, cells, agents, packets, particles, jobs), and what is the *complete* set of variables that defines one actor's situation right now? For a protocol node that is its phase, its current term/round, what it has voted for, what it believes about peers, its pending timers. Miss a state variable and the sim will lie.
- **The rules.** How does state change on each step? These must be the paper's actual rules, not a cartoon. Write the transition function in words before code.
- **Time model.** Is this continuous (forces, motion, flows integrated over small dt) or discrete-event (messages and timeouts that fire at specific virtual times) or both? This decides the engine in step 3.
- **The reader's questions.** List at least ten "what if" questions a curious reader asks. What if a node dies here? What if this message is delayed past the timeout? What if the network splits? What if two nodes both think they lead? What if I crank this parameter? Each question becomes either a parameter, an intervention, or a preset scenario. This list is your build spec.
- **Failure modes and edge cases from the paper.** These are the richest interventions. The paper's "we assume no more than f faults" is an invitation to let the reader add the (f+1)th and watch it break.
- **Invariants and properties.** What must always hold (safety: no two decisions disagree; conservation: count is preserved) and what should eventually happen (liveness: a decision is reached)? The sim must show these holding and, under the right intervention, breaking.
- **The emergent phenomenon.** What is the one thing worth watching unfold? Convergence, a cascade, an oscillation, a deadlock, a partition healing. The sim exists to make that visible and pokeable.
- **What must be observable.** List the internal state that has to be on screen for the phenomenon to make sense (beliefs per node, in-flight messages, queue depths, timers, phase per actor).

If you cannot fill this list, you do not understand the mechanism well enough to simulate it. Go back to the paper.

## Step 2. Architecture: split the model from the view

The single most important engineering decision. The model is headless and pure. The view is dumb and only draws.

- **Pure model.** A reducer `(state, input) => state` with no React, no DOM, no `Math.random`, no wall-clock. `input` is either a tick (advance virtual time by dt) or a discrete event/intervention. Because it is pure and deterministic, you can unit-test it and replay it.
- **Seeded randomness.** Thread a seeded PRNG through the state (mulberry32 or similar). Never call `Math.random`. The seed is a visible control. Same seed plus same inputs gives the same run, which is what makes "what just happened?" reproducible and scrubbing possible.
- **View.** React + SVG for tens of entities, canvas for hundreds, WebGL/three only when genuinely 3D. The view reads state and renders. It holds no rules.
- **Decouple update from frame rate.** Use a fixed-timestep accumulator on top of the shared `useAnimationFrame`. Accumulate elapsed ms, advance the model in fixed dt steps, render once per frame. This keeps behavior identical regardless of the device's frame rate.
- **History for scrub and rewind.** Keep a ring buffer of recent states (or of inputs, replayed from a seed). This buys step-back and "rewind to just before I killed it", which is most of the value of an intervention.

Sketch of the shapes:

```ts
type Rng = () => number; // seeded, in [0, 1)

function mulberry32(seed: number): Rng {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Input =
  | { kind: "tick"; dt: number }
  | { kind: "intervene"; action: Intervention };

// Pure. Deterministic given state (which carries the rng seed/cursor).
function step(state: SimState, input: Input): SimState;
```

## Step 3. Pick the engine

Most papers fall into one of two families. Many are a hybrid, and that is good.

**Discrete-event simulation** — for protocols, consensus, gossip, schedulers, queues, anything driven by messages and timeouts. The core is a priority queue of events keyed by virtual time. Each event mutates state and may schedule new events. Model the network explicitly: per-link latency, jitter, drop probability, and partitions. Each node is its own state machine with phases and timers. This family is what lets the reader kill a node between two phases:

```ts
interface Event {
  at: number; // virtual time
  to: NodeId;
  msg: Message; // e.g. { type: "accept", round, value }
}

// The reader pauses when node X enters `verified`, then fires:
function crash(state: SimState, id: NodeId): SimState {
  // mark the node dead; cancel its scheduled sends; deliver nothing further.
  // the `accept` it would have sent is never scheduled, peers' timers fire,
  // and quorum logic reacts on its own. you do not script the result.
}
```

**Continuous / physical** — for particles, forces, spatial layouts, flows. Use matter.js for real 2D physics (collisions, springs, gravity) or a small hand-rolled integrator when the forces are simple. Step in real time. Carry a spatial layer (positions, velocities) and a logical layer on top (each cell's beliefs). The bouncing cells are this: physics for the motion and legibility, gossip for the white-to-red truth spreading.

**Hybrid** — a discrete-event protocol laid out over a physical or force-directed canvas so messages visibly travel between moving nodes. Common for distributed-systems papers and usually the most legible.

Decide here, and write the model accordingly. The view follows.

## Step 4. Interventions are the product

A simulation without interventions is a presentation with extra steps. Build a real catalog. Each intervention is a user action, applied while the sim is running, that injects an event the model handles by its own rules.

By family, the usual suspects:

- **Distributed / protocol:** crash a node, restart it, drop or delay or duplicate or reorder a specific message, partition the network into groups and later heal it, make a node Byzantine (send conflicting messages), force a timeout, single-step one node's state machine, freeze one actor while others run.
- **Physical / agent:** drag an entity, add or remove entities, change a force or rate live, pin a value, inject a shock or impulse.
- **Universal:** change any parameter live, set the seed, jump to a preset scenario, slow time to line up a precise moment.

Non-negotiable rules for interventions:

- **Available mid-run, not only at setup.** The entire point is perturbing a system that is already going.
- **Targeted.** Click an entity to act on *that* one. Click an in-flight message to drop *that* message. Generic "add chaos" buttons are weak; precise strikes are the experience.
- **The model stays valid under any intervention at any time,** including ill-timed and adversarial ones. No crashes, no undefined stuck states. Whatever happens next is whatever the paper's rules produce. That honesty is the lesson. If killing the node deadlocks the protocol under a partition, it must deadlock on screen too.
- **Pause-aware aiming.** Let the reader pause, line up the intervention at the exact instant (the gap between `verify` and `accept`), then step or resume. Offer conditional breakpoints where useful ("pause the moment any node reaches `accept`") so the reader can hit a narrow window without reflexes.

## Step 5. Observability: show the inside, not just the outcome

The reader has to see why, not just what.

- **Render internal state directly.** Color or shape each actor by its belief or phase, always with a non-color cue too (label, icon, fill pattern) so it survives color blindness. Draw in-flight messages as moving tokens tagged with their type. Show queue depths and live timers.
- **Click to inspect.** A panel that dumps the full state of the selected entity (its phase, votes, what it believes about each peer, pending timers).
- **Event log / timeline.** A running, scrollable list of what happened (sent, delivered, dropped, timeout, crashed, decided). Make entries clickable to scrub the sim back to that moment.
- **Invariant indicators.** A visible light per property (safety: holding / violated; agreement reached: yes / no). When an intervention breaks safety, the reader watches the light flip. This is often the single most powerful element on screen.
- **A plot or counter for the emergent quantity** (how many nodes know the truth, percent in agreement, messages in flight over time).

## Step 6. Controls

play, pause, step (advance one event or one fixed tick), speed, scrub and rewind through history, reset, a seed field, parameter sliders, the intervention buttons plus click/drag targets, and a preset picker. Everything keyboard-operable. Reuse the shared `PlayPauseStepControls` and `useAnimationFrame` for the transport and extend from there; do not hand-roll a second clock.

## Step 7. Presets and scenarios

Named starting conditions, optionally with a short scripted sequence of interventions, that reproduce the paper's key results and its famous failure cases. "Happy path." "Leader crashes after prepare." "Network splits 3 / 2 then heals." "One Byzantine node among five." One click loads a preset. Presets are the guided narrative that respects agency: they aim the reader's attention at the phenomena that matter, and everything outside a preset is free exploration. This is how you guide without removing control.

## Fidelity and honesty

Implement the paper's real rules. Where you simplify (fewer nodes than reality, abstracted cryptography, an idealized network), say so in the UI with a short note. Never fake an outcome to look clean or to make a point the paper does not support. A simulation earns trust by behaving the way the paper says the system behaves, including the ugly parts.

## Robustness and engineering

- **Determinism and seed.** Same seed and same inputs reproduce the run exactly. No `Math.random`, no `Date.now` in the model.
- **Fixed timestep** so behavior is frame-rate independent.
- **Bounded work.** Cap entity counts and the event queue. Throttle. Do not re-render the whole world every tick; mutate per-tick state in refs and draw to canvas, or key React on coarse state, so the page stays smooth.
- **Client-only for canvas/WebGL.** Keep the component `"use client"` and draw inside a `useEffect` after mount, or lazy-load the heavy part with `next/dynamic` (`ssr: false`). Never draw during server render.
- **Error boundary.** The registry wraps every visualization in `VisualizationBoundary`, so a crash in the loop shows a fallback card. Keep it that way.
- **Tests on the pure model** (Vitest, next to the model file). Assert invariants hold on the happy path. Assert that specific interventions produce the expected consequence: killing a node between `verify` and `accept` yields no decision and a fired timeout; healing a partition lets a stalled round complete. These tests are cheap because the model is pure, and they are what prove the sim is honest.

## Abstraction timing (codebase rule)

The first simulation builds its own engine inline (its model, rng, event queue, history, inspector). Do not pre-build a generic framework from one example — that violates the codebase's abstraction-timing rule and you will guess the shape wrong. After the *second* simulation exists, look at what the two genuinely share and extract those parts into `lib/simulation/` (rng, fixed-timestep hook, event queue, history buffer) and `components/simulation/` (inspector shell, event-log panel, invariant lights). Two concrete cases first, then the abstraction.

## No meta-narration in the copy

When you write the words around the simulation (the "Try it" prompt, captions, on-screen hints), never describe the widget or announce what it is. Delete on sight: "This is a live model of the paper's core math, not a movie." "This is a simulation, not a slideshow." "Watch it run below." Show that it is live by telling the reader exactly what to do and what to watch, like "load the Leader crash preset, then kill node 3 right after it sends prepare and watch the round stall." The instruction proves it is interactive. The announcement only talks about it. The fidelity notes are different and still required, because they state plain facts about the model ("this runs 5 nodes, not 5000") rather than narrating the artifact. See clear-thinking-writing for the full rule.

## How this plugs into the pipeline

- **add-paper, "think it through" and "design the interactive":** run Step 1 here (the full enumeration) and decide the engine from Step 3. The output is a model spec (entities, state, rules, interventions, invariants, presets), not a list of animation stages.
- **add-paper, "write the visualization":** build the pure model first, test it, then the view and controls, then wire interventions. Register and boundary-wrap as usual.
- **paper-analyst, `<visualizing>`:** its interactive guidance defers here. Static-figure guidance stays in paper-analyst.
- **The doc's "Try it" section:** becomes a sandbox with the controls and a one-line prompt pointing the reader at a preset and a specific intervention to try first ("load Leader crash, then kill node 3 right after it sends prepare"). Write that prompt as a direct instruction, never as narration about the widget (see "No meta-narration in the copy" above).

## Definition of done

- There is a pure, seeded, deterministic model with unit tests, separate from the view.
- The reader can change a parameter and inject at least one targeted intervention while it runs, and the outcome changes accordingly under the paper's rules.
- The internal state is observable: entities encode their state, in-flight events are visible, there is an event log and at least one invariant indicator or emergent-quantity readout.
- Controls cover play, pause, step, speed, reset, seed, and scrub/rewind.
- At least two presets exist, one happy path and one failure case from the paper.
- It is client-only where it touches canvas/WebGL, wrapped in the error boundary, and `pnpm lint`, `pnpm typecheck`, `pnpm test`, and `pnpm build` all pass.
- If this is the second simulation in the repo, the shared parts have been extracted to `lib/simulation/` and `components/simulation/`.
