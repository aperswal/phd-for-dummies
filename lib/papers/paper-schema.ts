import { z } from "zod";

// Every figure carries non-empty alt text. An empty string fails validation, so
// the accessibility and SEO rule "images must describe what they show" is
// enforced at the data boundary rather than left to a reviewer to catch.
export const paperImageSchema = z.object({
  src: z.string().min(1),
  alt: z.string().min(1),
  width: z.number().int().positive().optional(),
  height: z.number().int().positive().optional(),
  caption: z.string().optional(),
});

// The contract for a paper's meta.json. The abstract lives here because it is
// what the sidebar search matches against. The slug must be kebab-case and, as
// get-papers enforces, must equal the folder name.
export const paperMetaSchema = z.object({
  slug: z
    .string()
    .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "slug must be kebab-case"),
  title: z.string().min(1),
  headline: z.string().min(1),
  abstract: z.string().min(1),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "date must be YYYY-MM-DD"),
  tags: z.array(z.string().min(1)).default([]),
  source: z
    .object({ name: z.string().min(1), url: z.string().url() })
    .optional(),
  images: z.array(paperImageSchema).default([]),
  hasVisualization: z.boolean().default(false),
});

export type PaperImage = z.infer<typeof paperImageSchema>;
export type PaperMeta = z.infer<typeof paperMetaSchema>;
