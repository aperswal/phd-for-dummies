---
name: paper-images
description: Design and generate the figures for a paper's explainer — beautiful, minimalist 2K images that carry both the concept and the everyday analogy, with no more than 3 to 4 elements each. Use from add-paper, or directly, whenever creating diagrams for a doc. Covers the one-message rule, the element budget, fusing the mechanism with the explainer's running analogy, the house style (an inked pixel-art field-journal look on aged parchment, a tight earthy colorblind-safe palette, short black serif labels), prompt craft for the Gemini image tool, aspect ratios, descriptive alt text, the generate command, and the look-at-it-then-regenerate quality loop.
---

# paper-images

You make figures that are a pleasure to look at and that teach. Each image carries one idea, shows it with no more than 3 to 4 elements, and fuses the technical concept with the everyday analogy the explainer already uses, so the picture and the words reinforce each other instead of competing. Minimalist, not sparse. Beautiful, not decorated. The reader should get the idea at a glance and want to keep looking.

The failure modes you are avoiding: a cluttered diagram with a dozen boxes, a wall of tiny labels the image model garbled, a dry schematic that carries the mechanism but none of the intuition, and stock-illustration noise (3D bevels, gradients for show, busy backgrounds) that competes with the content.

## Step 1. One message per image

State the single thing the image must land, in one sentence, before you design anything. If you can't say what it shows, you are not ready to generate it. "Attention lets every word look at every other word at once." "Gradient descent is a ball rolling to the bottom of a valley." One sentence, one image. If you have two messages, you have two images.

## Step 2. Fuse the concept with the analogy

The richest figures show the mechanism *and* the intuition in one frame. The analogy is not yours to invent here — pull the running analogy the explainer already chose (the ELI5 and mental-model work in paper-analyst and clear-thinking-writing) so the figure and the prose share one mental model. Three ways to combine them, in order of preference:

- **Fused (best).** One composition that is both at once. Attention as glowing lines between words that read like a conversation. A failure detector as ripples of color spreading through a crowd. Self-attention heads as colored lenses laid over a sentence.
- **Concept, analogy-flavored.** A precise mechanism diagram rendered in the analogy's visual language (the valley-shaped axis for gradient descent, the relay-baton motif for a sequential pipeline).
- **Side by side.** The analogy on one side, the mechanism on the other, only when fusing would muddle both. This still counts against the element budget.

Evoke the analogy through form, not clip-art. Build it from pixel-art objects and ink annotations in the house style below: a pixel boulder in a pixel valley with an inked path arrow, ripples spreading through a pixel crowd, lenses sketched in ink over a row of pixel word tiles. Not a cartoon mascot, not a photograph.

## Step 3. The element budget: 3 to 4, hard cap

Before generating, list the distinct visual elements the eye will parse. Keep it to 3 or 4. An "element" is a visual group, not a glyph: a row of five word tiles is one element ("a sentence"), a stack of vectors is one element, a fan of lenses is one element. If your list runs past four, you are trying to say too much in one image. Split it into two figures, drop to a higher level of abstraction, or move the detail into the MDX prose and code. One clear focal point, the rest supporting it.

## The house style: an inked pixel field-journal

Every figure looks like a page from an explorer's field journal in a cozy pixel game. Two layers fused on aged paper: a pixel-art scene that carries the analogy, and hand-inked annotations that carry the mechanism. The warm editorial restraint of ink on parchment meets the cozy nostalgic craft of Stardew Valley, Terraria, and Minecraft. Hold this across a paper's whole set and across papers.

- **Background:** aged cream parchment with faint paper grain and a soft torn or deckled edge. Warm, never pure white, never dark or busy. The parchment is the page, and the subject sits on it with generous margins.
- **Two layers.** Render the concept's objects as clean pixel art: chunky clean pixels, a limited palette, soft warm golden-hour light, and gentle dithered shading instead of smooth gradients. Then annotate over them with thin black ink sketch lines, small hand-drawn arrows, and braces, as if someone sketched over the scene in a notebook. The pixel layer is the intuition, the ink layer is the diagram. A strong pattern is a pixel scene set as a framed panel on the page with the ink labels and arrows reaching out into the margin.
- **Palette:** tight and earthy. Warm cream parchment, near-black ink, and a single clay-orange accent for the focal element, plus at most one muted secondary (sage green or dusty slate) for a second category. Every color is also distinct by shape or position, so meaning survives color blindness. Never encode meaning in color alone, and never red and green alone.
- **Forms:** clean pixel art plus deliberate ink linework. Dithering for shade, not glossy gradients. No photorealistic 3D, no glassy bevels, no neon glow, no busy background.
- **Hierarchy:** one element, usually the clay-orange one, is clearly the focal point. The eye knows where to land first, and the ink arrows lead it there.

## Text: short serif labels, still minimal

Black serif labels are part of this look, but image models still garble long text, so keep them spare:

- Use **zero to three labels**, each **1 to 2 words**, in a black Times New Roman serif, placed like journal annotations (a noun for the focal object, a verb on an arrow like "descent" or "attend").
- Short single tokens survive. Phrases and full sentences come out misspelled, so don't risk them.
- **Never** put a sentence, a paragraph, an equation, or a citation in the image. Those live in the MDX, where they render correctly and stay searchable. The picture carries intuition, the prose carries precision.

## Prompt craft

Write a descriptive natural-language prompt, not a keyword soup. Name the subject, the analogy fusion, the 3 to 4 elements and how they relate, then the house style, then what to avoid. A working template:

```
A cozy pixel-art illustration on an aged cream parchment background with faint
paper grain and a soft torn edge, annotated like a hand-inked field journal,
showing <the one message>, depicted as <the analogy fused with the concept>.
Render the subject as clean limited-palette pixel art with soft warm golden-hour
light and gentle dithered shading, in the cozy look of Stardew Valley and
Terraria, then draw the diagram over it as thin black ink sketch lines and small
hand-drawn arrows. Show only <N> elements: <element 1>, <element 2>,
<element 3>, and how they relate (<the flow or relationship>). Tight earthy
palette of warm cream parchment, near-black ink, and a single clay-orange accent
<plus one muted sage if a second category is needed>, every color also distinct
by shape so meaning survives color blindness. Generous margins, one clear focal
point. No realistic 3D, no glossy gradients, no busy background, no logos or
watermarks. Minimal text: at most <k> short black serif labels of one or two
words in a Times New Roman style; no sentences and no equations.
```

Keep the prompt to a few sentences. A longer prompt does not mean a better image; it means a busier one.

## Aspect ratio and resolution

- **16:9** for the hero and wide concept diagrams.
- **4:3** or **1:1** for inline diagrams.
- **3:4 / 9:16** only for genuinely tall subjects.

Generate at **2K**. The allowed aspects are `1:1`, `16:9`, `9:16`, `4:3`, `3:4`; resolutions are `1K`, `2K`, `4K`.

## Alt text

Every figure needs descriptive alt text, never empty, never a filename. Write a sentence that says what the figure shows and the takeaway, and let it carry the analogy too, so a screen-reader user gets the same intuition. It doubles as the Open Graph description for the hero. Name what is in the frame directly. Don't open with meta-narration about the artifact like "This image shows," "A diagram of," or "An illustration depicting." Start with the content itself, the way the example does. The same holds for the caption in meta.json. A caption states the idea in the picture, it never says "figure showing X." Example: "A short sentence of word tiles connected by thin curved lines of different thickness, like people in a conversation paying more attention to some speakers than others, showing that attention weights every word against every other word at once."

## Generate

Make the output directory, then call the tool once per figure. It runs Gemini 3 Pro Image, takes a custom `--output`, and prints the saved path:

```bash
mkdir -p public/papers/<slug>
Script-Image-Gen/.venv/bin/python Script-Image-Gen/generate.py \
  --prompt "<the prompt>" \
  --aspect <ratio> \
  --resolution 2K \
  --output "$(pwd)/public/papers/<slug>/<id>.png"
```

The `src` in `meta.json` and in the markdown is the public path (`/papers/<slug>/<id>.png`), not the filesystem path.

## Look at it, then regenerate if needed

Always open each generated image with the Read tool and judge it against this rubric:

- One message, landing at a glance.
- 3 to 4 elements, not more. One clear focal point.
- The analogy is present in the pixel layer, not just a dry schematic.
- Text is minimal and **not garbled**. No misspelled words, no fake glyphs. Labels are black serif.
- Aged parchment background, pixel-art subject with dithered shading, hand-inked annotations on top.
- Earthy palette (cream, ink, clay-orange, maybe one sage), generous margins, one clear focal point.
- No photoreal 3D, no glossy gradients, no watermark or logo.
- It is genuinely nice to look at.

If it fails any line, adjust the prompt and regenerate. The usual fixes: cut an element, remove a label that came out garbled, push more whitespace, simplify the composition, or restate the analogy more concretely. Budget one or two regenerations per figure. Keep the best result and delete the rejects.

## Consistency across the set

A paper's figures are a family. Reuse the same parchment, the same pixel resolution and earthy palette, and the same ink-annotation treatment across the hero and the inline diagrams, so the page reads as one journal from one pixel world rather than a pile of assembled images. If the explainer's analogy has a recurring motif (lenses, ripples, a valley), echo it across the set.

## How this plugs into the pipeline

- **add-paper, "design the figures":** runs Steps 1 to 3 here (one message, fuse the analogy, count the elements) for the hero plus one to three inline diagrams, scoped to this doc only, never a global library.
- **add-paper, "generate the figures":** runs the command above and the look-at-it-then-regenerate loop before moving on.
- **paper-analyst, static figures:** its data-ink and chunking principles still hold for analytic charts; generated explainer images follow this skill's tighter element budget, analogy fusion, and house style.

## Definition of done

- Each figure carries one message with 3 to 4 elements and the explainer's analogy.
- The set shares one parchment, earthy palette, pixel style, and ink-annotation treatment; margins are generous.
- Text is minimal, black serif, and clean, with no garbled words; sentences and equations stayed in the MDX.
- Every figure has descriptive, non-empty alt text that also conveys the intuition.
- Each image was opened and judged, and anything cluttered, off-message, or garbled was regenerated.
