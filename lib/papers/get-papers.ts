import "server-only";

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

import { logger } from "@/lib/logger";
import { paperMetaSchema, type PaperMeta } from "@/lib/papers/paper-schema";
import type { PaperListItem } from "@/lib/search/types";

const PAPERS_DIR = join(process.cwd(), "content", "papers");

// Reads and validates every paper's meta.json. A malformed file throws rather
// than getting skipped, so bad metadata fails the build instead of silently
// dropping a paper from the site. Returned newest first.
export async function getAllPapers(): Promise<PaperMeta[]> {
  let entries;
  try {
    entries = await readdir(PAPERS_DIR, { withFileTypes: true });
  } catch {
    return [];
  }

  const papers: PaperMeta[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;

    const metaPath = join(PAPERS_DIR, entry.name, "meta.json");
    let raw: string;
    try {
      raw = await readFile(metaPath, "utf8");
    } catch {
      logger.warn(
        { dir: entry.name },
        "paper folder has no meta.json, skipping",
      );
      continue;
    }

    const parsed = paperMetaSchema.safeParse(JSON.parse(raw));
    if (!parsed.success) {
      logger.error(
        { dir: entry.name, issues: parsed.error.issues },
        "invalid meta.json",
      );
      throw new Error(
        `Invalid meta.json in content/papers/${entry.name}: ${parsed.error.message}`,
      );
    }
    if (parsed.data.slug !== entry.name) {
      throw new Error(
        `slug "${parsed.data.slug}" does not match folder "${entry.name}"`,
      );
    }
    papers.push(parsed.data);
  }

  return papers.sort((a, b) => (a.date < b.date ? 1 : -1));
}

// The slim list the client sidebar searches over.
export async function getPaperListItems(): Promise<PaperListItem[]> {
  const papers = await getAllPapers();
  return papers.map(({ slug, title, abstract }) => ({ slug, title, abstract }));
}
