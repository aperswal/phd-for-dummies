"use client";

import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

// One card's worth of a paper. The grid shows the title and headline only; tags
// stay out of the home view to keep each card to a glanceable two lines.
export interface PaperCard {
  slug: string;
  title: string;
  headline: string;
}

const PER_PAGE = 6;

// A paginated grid of paper cards. It shows one page of six at a time so the
// list fits without scrolling, and pages through the rest with numbered
// controls. The page index is local because it isn't worth sharing in the URL.
export function PaperGrid({ papers }: { papers: PaperCard[] }) {
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(papers.length / PER_PAGE));
  const current = Math.min(page, pageCount - 1);
  const start = current * PER_PAGE;
  const visible = papers.slice(start, start + PER_PAGE);

  return (
    <div className="flex flex-col gap-6">
      <ul className="grid gap-4 sm:grid-cols-2">
        {visible.map((paper) => (
          <li key={paper.slug}>
            <Link href={`/papers/${paper.slug}`} className="block h-full">
              <Card className="hover:border-foreground/30 h-full gap-2 p-6 transition-colors">
                <h2 className="text-lg leading-snug font-medium">
                  {paper.title}
                </h2>
                <p className="text-muted-foreground line-clamp-3 text-sm">
                  {paper.headline}
                </p>
              </Card>
            </Link>
          </li>
        ))}
      </ul>

      {pageCount > 1 && (
        <nav
          aria-label="Pagination"
          className="flex flex-wrap items-center justify-center gap-1.5"
        >
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setPage(current - 1)}
            disabled={current === 0}
            aria-label="Previous page"
          >
            Prev
          </Button>
          {Array.from({ length: pageCount }, (_, index) => (
            <Button
              key={index}
              type="button"
              size="sm"
              variant={index === current ? "default" : "outline"}
              aria-label={`Page ${index + 1}`}
              aria-current={index === current ? "page" : undefined}
              onClick={() => setPage(index)}
            >
              {index + 1}
            </Button>
          ))}
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setPage(current + 1)}
            disabled={current === pageCount - 1}
            aria-label="Next page"
          >
            Next
          </Button>
        </nav>
      )}
    </div>
  );
}
