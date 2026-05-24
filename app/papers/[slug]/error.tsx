"use client";

import Link from "next/link";

import { Button } from "@/components/ui/button";

// Outer safety net for the paper route. The per-visualization boundary handles
// a crashing interactive on its own; this only fires if loading or rendering
// the page itself fails.
export default function PaperError({ reset }: { reset: () => void }) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-20 text-center">
      <h1 className="text-2xl font-semibold">
        Something broke loading this paper
      </h1>
      <p className="text-muted-foreground mt-2">
        Try again, or head back to the full list.
      </p>
      <div className="mt-6 flex justify-center gap-3">
        <Button type="button" onClick={reset}>
          Try again
        </Button>
        <Button variant="secondary" render={<Link href="/" />}>
          All papers
        </Button>
      </div>
    </div>
  );
}
