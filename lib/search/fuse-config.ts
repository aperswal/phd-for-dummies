import Fuse, { type IFuseOptions } from "fuse.js";

import type { PaperListItem } from "@/lib/search/types";

// Title outweighs abstract so a title hit ranks first. ignoreLocation is on
// because abstracts are long and the default position penalty would bury a
// match that sits mid-paragraph. threshold 0.4 balances fuzzy recall against
// noise.
export const fuseOptions: IFuseOptions<PaperListItem> = {
  keys: [
    { name: "title", weight: 0.7 },
    { name: "abstract", weight: 0.3 },
  ],
  ignoreLocation: true,
  threshold: 0.4,
  minMatchCharLength: 2,
};

export function buildPaperIndex(papers: PaperListItem[]): Fuse<PaperListItem> {
  return new Fuse(papers, fuseOptions);
}
