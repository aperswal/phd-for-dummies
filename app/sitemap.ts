import type { MetadataRoute } from "next";

import { env } from "@/lib/env";
import { getAllPapers } from "@/lib/papers/get-papers";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const base = env.NEXT_PUBLIC_SITE_URL;
  const papers = await getAllPapers();
  return [
    { url: base, lastModified: new Date() },
    ...papers.map((paper) => ({
      url: `${base}/papers/${paper.slug}`,
      lastModified: new Date(paper.date),
    })),
  ];
}
