import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

// A titled bordered panel for a simulation's side sections (controls, inspector,
// readouts), so every panel across visualizations shares one frame and spacing.
export function SimPanel({
  title,
  children,
  className,
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "ring-foreground/10 flex flex-col gap-3 rounded-xl p-4 ring-1",
        className,
      )}
    >
      <span className="text-muted-foreground text-xs font-medium">{title}</span>
      {children}
    </div>
  );
}
