---
name: add-paper
description: Turn a research paper PDF into a finished entry on this site. Use when the user drops a PDF into papers/ and runs /add-paper papers/<file>.pdf, or @-mentions this skill plus a PDF on a fresh session. Runs the whole pipeline end to end — read the paper, think it through, break it down, write image specs, generate 2K diagrams, write a layered explainer, build a bespoke play/step visualization, register it, and build the project — so a new searchable entry appears in the sidebar with a working page.
---

# add-paper

You take one research paper PDF and produce a complete, building, searchable entry on this Next.js site. You run every step yourself, in order, on the given PDF path. You read and apply two other skills as you go: **paper-analyst** (how to read, break down, and visualize a paper) and **clear-thinking-writing** (how to reason through it and how to write the prose).

## How you're invoked

The user runs one of these on a fresh session:

```
/add-paper papers/<file>.pdf
```

or `@.claude/skills/add-paper/SKILL.md @papers/<file>.pdf`. Either way you get a path to a PDF sitting in `papers/`. If the path is missing or isn't a PDF, stop and ask for it. Do not guess.

## The four slug-keyed touch points

Adding one paper writes to exactly these locations, all keyed by the same kebab-case `<slug>`:

1. `content/papers/<slug>/meta.json` and `content/papers/<slug>/doc.mdx` — the metadata (drives the sidebar and search) and the explainer.
2. `public/papers/<slug>/<id>.png` — the generated 2K diagrams.
3. `components/visualizations/<slug>.tsx` — the bespoke interactive visualization.
4. One import and one entry in `components/visualizations/registry.tsx` — wires the visualization to the page.

The slug must be kebab-case and must equal the folder name under `content/papers/`. `getAllPapers` throws if they disagree, so a mismatch fails the build, not the page.

## The pipeline

Run these steps in order. Do not skip the codebase traversal — it's what keeps every new paper matching the existing pattern instead of inventing a new one.

### 0. Load the skills

Read `.claude/skills/paper-analyst/SKILL.md` and `.claude/skills/clear-thinking-writing/SKILL.md` in full before you start. They govern how you read the paper, how you break it down, what the visualization must do, and how every sentence in the explainer is written.

### 1. Traverse the codebase

Read these so the new entry matches the existing shape:

- `lib/papers/paper-schema.ts` — the exact `meta.json` contract (field names, the kebab-case slug rule, non-empty alt text, the `images[]` shape, `hasVisualization`).
- `lib/papers/get-papers.ts` — how papers are discovered and validated, and the slug-equals-folder rule.
- `app/papers/[slug]/page.tsx` — how the page injects `<Visualization />` and renders the MDX inside `prose`.
- `components/mdx/mdx-renderer.tsx` and `components/mdx/mdx-components.tsx` — the remark/rehype pipeline and which elements are overridden.
- `components/visualizations/registry.tsx` — the slug→component map you'll add to.
- `components/visualizations/play-pause-step-controls.tsx`, `components/visualizations/use-animation-frame.ts`, `components/visualizations/visualization-boundary.tsx` — the shared transport, frame loop, and error boundary every visualization uses.
- If a `content/papers/*/` folder already exists, read its `meta.json` and `doc.mdx` and copy the pattern.

### 2. Read the paper

Read the PDF with the Read tool in the **paper-analyst reading protocol order**: abstract, figures, conclusion, results, methods, introduction, related work. PDFs over 10 pages need the `pages` parameter; read in batches.

Extract the **title** and the **abstract** (the abstract is what the sidebar search matches on, so keep it real, not a paraphrase). Derive the **slug** as kebab-case of the title (drop articles if it's long, e.g. `attention-is-all-you-need`). Find the arXiv / DOI / publisher URL for `source`.

### 3. Think it through

Run **clear-thinking-writing**'s seven lenses on the paper. The goal of this pass is two answers:

- The **one core mechanism** — the single equation, algorithm, or insight everything else supports. If you can't state it in one sentence, you're not done reading.
- **What the reader should be able to poke.** The interactive is a simulation the reader perturbs, not a slideshow. Decide whether the mechanism is a dynamical system, protocol, algorithm, or multi-agent process the reader can intervene in (kill a node mid-protocol, drop a message, partition the network, change a rate and watch it cascade). If it is, the **paper-simulation** skill governs the design and you do the full think-first enumeration there. If the mechanism genuinely is a one-way transformation with nothing meaningful to perturb, a lighter staged explainer is acceptable, but still add what interactivity you can (edit the input, click to inspect, mask a part and watch the output change). Either way, pick the technology that matches how the mechanism actually behaves: real 2D physics with matter.js, 3D with three/react-three-fiber, discrete-event message passing built on an event queue, staged motion with framer-motion, or data-driven views with d3.

### 4. Break it down

Run **paper-analyst**'s summarizing flow: the headline (one sharable jargon-free sentence), the executive summary (problem, approach, result, limitation), the structured extraction (problem and why prior work failed, key idea, methodology, results with effect sizes, limitations, open questions), and your assessment.

Then design the interactive by following the **paper-simulation** skill. Produce a model spec, not a list of animation stages: the entities and their full state, the paper's real transition rules, the engine (discrete-event, physical, or hybrid), the intervention catalog (what the reader can do mid-run and what each does), the invariants the sim shows holding or breaking, the things that must be observable, and at least two presets (a happy path and a failure case from the paper).

### 5. Design the figures

Design the diagrams this one paper needs by following the **paper-images** skill. Not a global library — only what reduces confusion on this page, usually one hero plus one to three inline diagrams. For each, do the paper-images think-first work: one message, fuse the mechanism with the explainer's running analogy, and count the elements (3 to 4 max). Each spec carries:

- an `id` (kebab-case, becomes the filename, e.g. `hero`, `attention-heatmap`),
- a one-sentence **key message**,
- the **prompt** written per paper-images (minimalist flat-vector house style, the analogy fusion, the 3-to-4-element budget, almost no text),
- an **aspect ratio** (`16:9` for hero/wide, `1:1` or `4:3` for inline),
- **descriptive alt text** that says what the figure shows, the takeaway, and carries the analogy — never empty, never just a filename.

### 6. Write meta.json

Create `content/papers/<slug>/meta.json` matching `paperMetaSchema` exactly:

```json
{
  "slug": "<slug>",
  "title": "<paper title>",
  "headline": "<one sharable sentence>",
  "abstract": "<the real abstract — what search matches on>",
  "date": "<YYYY-MM-DD, today>",
  "tags": ["<topic>", "<method>"],
  "source": { "name": "arXiv 1706.03762", "url": "https://arxiv.org/abs/1706.03762" },
  "images": [
    {
      "src": "/papers/<slug>/hero.png",
      "alt": "<descriptive alt text>",
      "width": 2048,
      "height": 1152,
      "caption": "<optional one-line caption>"
    }
  ],
  "hasVisualization": true
}
```

Writing this file is what adds the paper to the sidebar and to fuse.js search. There is no separate list to edit. The hero image (first in `images[]`) is also the Open Graph image, so its alt text matters for sharing.

### 7. Write the explainer (doc.mdx)

Write `content/papers/<slug>/doc.mdx` using the **layered template** below. Apply **clear-thinking-writing**'s voice to every sentence (active voice, plain words, contractions, build from the building blocks up so each idea lands before the one that needs it, the simpler word whenever one carries the meaning, no filler, no em dashes, no colons inside sentences), and the **paper-analyst** structure for the document as a whole. Per the precedence note in clear-thinking-writing, headings, fenced code blocks, and tables are correct here — the no-headers rule is for Slack, not this page.

Write no meta-narration. Never describe the page, a section, a figure, or the interactive, and never announce structure. The "Try it" copy is the usual trap. Write a direct instruction like "Load the saturation preset, then drag the temperature up and watch the weights spread," never "This is a live model, not a movie." When you name where an analogy breaks, state the real mechanism as a fact, not as a comment on the picture. See clear-thinking-writing for the banned examples.

Gotchas that will silently break the page if you miss them:

- **Never write `import` statements inside the MDX.** The MDX is string-compiled by `next-mdx-remote/rsc` and can't resolve imports. The visualization reaches the page through the registry, injected as a bare `<Visualization />` element. Just write `<Visualization />` where the interactive goes.
- Reference images with standard markdown: `![<descriptive alt text>](/papers/<slug>/<id>.png)`. The alt text in the markdown is the real alt text — keep it descriptive, never empty.
- Fence pseudocode with a language tag (```python, ```text) so `rehype-pretty-code` highlights it at build time.

### 8. Generate the figures

Generate per the **paper-images** skill. The tool has its own `.venv` and a working `GEMINI_API_KEY` in `Script-Image-Gen/.env`, produces 2K PNGs, takes a custom `--output`, and prints the saved path:

```bash
mkdir -p public/papers/<slug>
Script-Image-Gen/.venv/bin/python Script-Image-Gen/generate.py \
  --prompt "<the prompt>" \
  --aspect <ratio> \
  --resolution 2K \
  --output "$(pwd)/public/papers/<slug>/<id>.png"
```

Then run the paper-images quality loop: open each generated image with the Read tool, judge it against the rubric (one message, 3 to 4 elements, the analogy present, text minimal and not garbled, house-style palette, genuinely nice to look at), and regenerate with an adjusted prompt if it falls short. Keep the best result, delete the rejects. The `src` in `meta.json` and in the markdown is the public path (`/papers/<slug>/<id>.png`), not the filesystem path.

### 9. Build the simulation

Follow the **paper-simulation** skill. Build the pure model first, then test it, then the view. Concretely:

- Write the headless, deterministic model as a `(state, input) => step` reducer with a seeded RNG, separate from any React. Put it next to the component (e.g. `components/visualizations/<slug>-model.ts`).
- Write unit tests for the model: invariants hold on the happy path, and each intervention produces the consequence the paper predicts (kill a node between two phases → no decision, timeout fires).
- Write `components/visualizations/<slug>.tsx` as a `"use client"` view that renders the model's state and drives it through the shared `PlayPauseStepControls` and `useAnimationFrame` (don't hand-roll a second transport) on a fixed timestep.
- Wire the interventions from your step-4 catalog as mid-run controls (buttons plus click/drag targets), add the inspector, event log, and at least one invariant indicator or emergent-quantity readout, and add the presets.
- Render canvas/WebGL client-only. Either keep it `"use client"` and draw inside a `useEffect` after mount, or lazy-load the heavy part with `next/dynamic` (`ssr: false`). Don't draw to canvas during server render.

Export a named component. If this is the second simulation in the repo, first extract the shared engine parts (rng, fixed-timestep hook, event queue, history buffer, inspector and event-log shells) into `lib/simulation/` and `components/simulation/` per the paper-simulation skill's abstraction-timing note, then build on them.

### 10. Register it

Add the import and one entry to `components/visualizations/registry.tsx`, wrapping the component in the shared error boundary so a crash in its loop shows a fallback card instead of taking down the page:

```tsx
import type { ComponentType } from "react";

import { VisualizationBoundary } from "@/components/visualizations/visualization-boundary";
import { AttentionFlow } from "@/components/visualizations/attention-is-all-you-need";

const visualizations: Record<string, ComponentType> = {
  "attention-is-all-you-need": () => (
    <VisualizationBoundary>
      <AttentionFlow />
    </VisualizationBoundary>
  ),
};

export function getVisualization(slug: string): ComponentType | null {
  return visualizations[slug] ?? null;
}
```

If the visualization needs a library that isn't already installed, install it with pnpm. The toolkit installed up front is `matter.js`, `three`, `@react-three/fiber`, `@react-three/drei`, `framer-motion`, and `d3` (with their `@types`). Only add a new dependency if the chosen approach genuinely needs one.

### 11. Build and report

Run, in order, and fix until each passes:

```bash
pnpm lint
pnpm typecheck
pnpm build
```

The React Compiler ESLint rules are strict — no ref writes during render, no setState in effects without the hydration-safe pattern (see `paper-sidebar.tsx` and `use-animation-frame.ts` for the patterns that pass). When all three are green, report the new route `/papers/<slug>`, confirm the sidebar entry and search both find it, and confirm the simulation runs, a preset loads, and at least one mid-run intervention changes the outcome.

## The layered doc template

Each `doc.mdx` reads top to bottom as a ramp, with a jump-nav near the top so a reader skips to their level. Headings get anchor ids from `rehype-slug`, so everything stays in the DOM (good for search and SEO) and the jump links work. Order:

1. `# Title` then a one-line italic headline.
2. The hero diagram: `![<alt>](/papers/<slug>/hero.png)`.
3. **Read at your level** — a short list of links to the level sections below (`[For a 5-year-old](#for-a-5-year-old)`, ... `[For a peer researcher](#for-a-peer-researcher)`).
4. `## Executive summary` — one paragraph (problem, approach, result, limitation).
5. `## Try it` — the bare `<Visualization />` (a sandbox the reader pokes) with one or two sentences pointing the reader at a preset and a specific intervention to try first, e.g. "Load the leader-crash preset, then kill node 3 right after it sends prepare and watch the round stall." Make those sentences direct instructions, never narration about the widget.
6. The level ladder, each its own `##` heading so the jump-nav anchors resolve, ramping up:
   - `## For a 5-year-old`
   - `## For a high schooler`
   - `## For a college student`
   - `## For an industry pro`
   - `## For a PhD candidate`
   - `## For a peer researcher`
   Each section obeys that audience's constraints from paper-analyst's `<audience_levels>`. The heading names the level, so start each section straight into the explanation. Do not add an italic "who it's for" note under the heading.
7. `## How it works` — the structured extraction (problem and why prior work failed, the key idea, methodology, results with effect sizes, limitations, open questions), with fenced code blocks where the mechanism is clearest as pseudocode, and inline diagrams placed exactly where they cut confusion.
8. `## My assessment` — your judgment: what they got right, what they got wrong, what's next.

Place the generated diagrams where they reduce confusion, not all at the top. The reader should never look back and forth between a figure and the text that explains it — keep them adjacent.

## What "done" looks like

The new entry shows in the sidebar. Lexical search finds it by a word in the title and by a word in the abstract. `/papers/<slug>` renders the layered explainer with syntax-highlighted code and 2K diagrams that load with real alt text. The visualization plays and steps, and forcing an error in it shows the fallback card while the rest of the page stays up. `pnpm lint`, `pnpm typecheck`, and `pnpm build` all pass.
