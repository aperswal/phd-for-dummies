"use client";

import type { ReactNode } from "react";
import { ErrorBoundary, type FallbackProps } from "react-error-boundary";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

function VisualizationFallback({ resetErrorBoundary }: FallbackProps) {
  return (
    <Card className="flex flex-col items-start gap-3 p-6">
      <p className="text-muted-foreground text-sm">
        This interactive demo couldn&apos;t load. The rest of the page is fine.
      </p>
      <Button
        type="button"
        size="sm"
        variant="secondary"
        onClick={resetErrorBoundary}
      >
        Try again
      </Button>
    </Card>
  );
}

// Wraps a single visualization so a crash in its render or physics loop shows a
// fallback card right here, keeping the surrounding prose, images, and sidebar
// alive. This is the per-widget blast radius the route-level error.tsx is too
// coarse to give.
export function VisualizationBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary FallbackComponent={VisualizationFallback}>
      {children}
    </ErrorBoundary>
  );
}
