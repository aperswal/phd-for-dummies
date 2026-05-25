# PhD for Dummies

Famous research papers, explained in layers, with diagrams and a live demo you can break.

Most people never read a research paper. Not because they can't think. Because the paper was written for the few hundred people already in the field. You open one, hit a wall of symbols on page 1, and close it. The idea inside is often simple. The writing buries it.

This site digs the idea back out. It takes a famous paper and explains it at every level, one layer at a time. The bottom layer is the version a 5 year old could follow. Then you climb. High schooler, college student, working engineer, PhD student, and at the top the version a peer researcher would argue with. You step off whenever you have what you came for. Want to know what "attention" means? The second floor is enough. Building it this week? Keep climbing.

Picture each paper as a tall building with the good stuff on the top floor and no elevator. The usual explanations drop you at one of two doors. The pop-science door leaves you in the lobby with a cute metaphor and nothing you can use. The paper itself opens straight onto the top floor with no stairs under your feet, so you fall. This site is the staircase. Every floor is a real place to stand, and each one is wired to the floor below, so the climb never skips a step.

Every page hands you the same three things. The write-up in those layers, so you read at your level instead of the author's. Diagrams drawn to carry one idea each, so a picture does the work a clumsy paragraph would. And a working model of the paper's core idea, sitting in the page, waiting for you to poke it.

That last part is the one most sites fake. They slap the word interactive on a slide show with a play button. You press play, shapes drift across the screen, and you learn nothing a gif wouldn't teach you. Here the demo runs the paper's real rules. You can reach in and break it. Kill a node in the middle of the search. Flip the reward from honest to gamed. Drag a knob past its safe range. The result changes, and it changes for the reason the paper gives, because the machinery underneath is the paper's machinery and not a cartoon of it. You come to understand a mechanism by messing with it, the way you learn a light switch by flipping it instead of reading about electricity.

The belief under all of this is plain. Understanding is not a thing you have or don't. It comes in heights. A good explanation meets you where you stand and hands you a way up. Most explanations pick one height and strand everyone else. This one refuses to pick.

Right now it holds 34 papers, from the reinforcement learning roots of the 1980s through the transformer, the large language models, and the agent work of the last few years. None of it is handwritten. Every word, every diagram, and every simulation is produced by an agent working from the paper's PDF, each one running the same fixed path. Read the paper. Find the single idea everything hangs on. Draw it. Explain it in layers. Build a model of it you can play with. The repo is really that harness, with its output checked in beside it.

## Run it locally

It's a Next.js site. You need Node and pnpm.

```bash
pnpm install
pnpm dev
```

Open http://localhost:3000. To run every check the build runs:

```bash
pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

One thing the repo leaves out is the diagrams. They are large image files, so they stay out of git. Clone this and the pages still explain and still run their demos, but the diagram spots come up blank until the images get regenerated. A separate offline tool draws them and is not part of this repo.

## How every page is built

Nothing on this site is written by hand. Each entry is produced by an agent running a fixed pipeline of skills over the paper's PDF, and the repo is that harness with its output checked in beside it. You drop a PDF into `papers/` and run one command:

```bash
/add-paper papers/your-paper.pdf
```

From there the agent reads the paper, finds the one idea everything hangs on, designs and generates the diagrams, writes the layered explainer, builds a real interactive model of the mechanism and tests it, reviews that model against a strict rubric, wires it into the page, and runs the build. What comes out is a finished, searchable entry in the sidebar with a working page. No step is mocked and no prose is pre-written.

The pipeline is six skills in `.claude/skills/`, each with one job:

- **add-paper** orchestrates the whole run, eleven steps from PDF to a passing build.
- **paper-analyst** reads the paper in figures-first order and breaks it into the headline, the summary, and the one core mechanism.
- **clear-thinking-writing** reasons through the paper and writes every sentence of the explainer in one plain voice, from the 5-year-old layer up to the peer-researcher one.
- **paper-images** designs and generates the 2K diagrams, one idea each, fused with the explainer's running analogy.
- **paper-simulation** builds the interactive as a deterministic, headless model with unit tests, then a client view on a shared transport, so the demo runs the paper's real rules.
- **simulation-review** audits that model against a UI, legibility, and correctness rubric and fixes what falls short.

What keeps unattended generation honest is the gates. Every simulation is a pure model with a test file beside it, every entry has to pass `pnpm lint && pnpm typecheck && pnpm test && pnpm build` before it counts as done, and a mismatched slug fails the build instead of shipping a broken page.
