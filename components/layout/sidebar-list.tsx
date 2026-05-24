"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";

import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { buildPaperIndex } from "@/lib/search/fuse-config";
import type { PaperListItem } from "@/lib/search/types";
import { cn } from "@/lib/utils";

interface SidebarListProps {
  papers: PaperListItem[];
  onNavigate?: () => void;
}

// The search box plus the filtered list, shared by the desktop aside and the
// mobile sheet. Fuse is built once per paper set; the query just filters it.
export function SidebarList({ papers, onNavigate }: SidebarListProps) {
  const [query, setQuery] = useState("");
  const pathname = usePathname();
  const fuse = useMemo(() => buildPaperIndex(papers), [papers]);

  const results = useMemo(() => {
    const trimmed = query.trim();
    if (!trimmed) return papers;
    return fuse.search(trimmed).map((result) => result.item);
  }, [query, fuse, papers]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="p-3">
        <Input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search papers"
          aria-label="Search papers by title or abstract"
        />
      </div>
      <ScrollArea className="min-h-0 flex-1 px-2 pb-4">
        {papers.length === 0 ? (
          <p className="text-muted-foreground px-2 py-6 text-sm">
            No papers yet. Drop a PDF into{" "}
            <code className="font-mono">papers/</code> and run{" "}
            <code className="font-mono">/add-paper</code>.
          </p>
        ) : results.length === 0 ? (
          <p className="text-muted-foreground px-2 py-6 text-sm">
            No papers match that search.
          </p>
        ) : (
          <nav aria-label="Papers">
            <ul className="space-y-1">
              {results.map((paper) => {
                const href = `/papers/${paper.slug}`;
                const active = pathname === href;
                return (
                  <li key={paper.slug}>
                    <Link
                      href={href}
                      onClick={onNavigate}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "hover:bg-accent hover:text-accent-foreground block rounded-md px-3 py-2 text-sm transition-colors",
                        active &&
                          "bg-accent text-accent-foreground font-medium",
                      )}
                    >
                      {paper.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        )}
      </ScrollArea>
    </div>
  );
}
