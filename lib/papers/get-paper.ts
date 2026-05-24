import "server-only";

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { paperMetaSchema, type PaperMeta } from "@/lib/papers/paper-schema";

const PAPERS_DIR = join(process.cwd(), "content", "papers");

export interface PaperContent {
  meta: PaperMeta;
  mdx: string;
}

// Loads one paper's validated metadata and raw MDX. Returns null when the slug
// has no folder so the route can render a 404 instead of throwing.
export async function getPaper(slug: string): Promise<PaperContent | null> {
  const dir = join(PAPERS_DIR, slug);
  let metaRaw: string;
  let mdx: string;
  try {
    [metaRaw, mdx] = await Promise.all([
      readFile(join(dir, "meta.json"), "utf8"),
      readFile(join(dir, "doc.mdx"), "utf8"),
    ]);
  } catch {
    return null;
  }
  const meta: PaperMeta = paperMetaSchema.parse(JSON.parse(metaRaw));
  return { meta, mdx };
}
