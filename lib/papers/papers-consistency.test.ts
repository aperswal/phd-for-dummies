import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { paperMetaSchema } from "@/lib/papers/paper-schema";

// Guards the slug-keyed touch points the add-paper pipeline writes to: a paper
// folder, its doc, its referenced images, and its visualization registration.
// A missing piece fails here at test time rather than as a broken page in prod.
// With no papers yet, the loop is empty and the test passes.
const PAPERS_DIR = join(process.cwd(), "content", "papers");
const VIZ_DIR = join(process.cwd(), "components", "visualizations");
const PUBLIC_DIR = join(process.cwd(), "public");

function paperFolders(): string[] {
  return readdirSync(PAPERS_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
}

describe("paper content matches the site", () => {
  const registry = readFileSync(join(VIZ_DIR, "registry.tsx"), "utf8");

  it("validates every paper folder against the site", () => {
    for (const folder of paperFolders()) {
      const meta = paperMetaSchema.parse(
        JSON.parse(readFileSync(join(PAPERS_DIR, folder, "meta.json"), "utf8")),
      );

      expect(meta.slug).toBe(folder);
      expect(existsSync(join(PAPERS_DIR, folder, "doc.mdx"))).toBe(true);

      for (const image of meta.images) {
        expect(image.alt.trim().length).toBeGreaterThan(0);
        // Figures are hosted on Vercel Blob, so the src is an absolute URL on
        // the store's public host. A local path (the pre-Blob form) must still
        // resolve to a file on disk.
        if (/^https?:\/\//.test(image.src)) {
          const { hostname } = new URL(image.src);
          expect(hostname.endsWith(".public.blob.vercel-storage.com")).toBe(
            true,
          );
        } else {
          expect(existsSync(join(PUBLIC_DIR, image.src))).toBe(true);
        }
      }

      if (meta.hasVisualization) {
        expect(existsSync(join(VIZ_DIR, `${folder}.tsx`))).toBe(true);
        // The slug must be a registry key. Prettier quotes hyphenated slugs and
        // leaves single-word ones bare, so match either form, anchored on a
        // boundary and a trailing colon so a slug can't match inside a longer one.
        const escaped = folder.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const keyPattern = new RegExp(`(?:^|[\\s{,])"?${escaped}"?\\s*:`, "m");
        expect(keyPattern.test(registry)).toBe(true);
      }
    }
  });
});
