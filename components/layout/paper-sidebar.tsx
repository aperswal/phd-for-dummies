"use client";

import { Menu, PanelLeft } from "lucide-react";
import Link from "next/link";
import { useCallback, useState, useSyncExternalStore } from "react";

import { SidebarList } from "@/components/layout/sidebar-list";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import type { PaperListItem } from "@/lib/search/types";
import { cn } from "@/lib/utils";

const COLLAPSE_KEY = "sidebar-collapsed";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

// Reads the collapsed preference from localStorage in a hydration-safe way: the
// server snapshot is always false, so the first client render matches the
// server HTML, then React reconciles to the stored value.
function usePersistentCollapsed(): [boolean, (next: boolean) => void] {
  const collapsed = useSyncExternalStore(
    subscribe,
    () => window.localStorage.getItem(COLLAPSE_KEY) === "1",
    () => false,
  );
  const setCollapsed = useCallback((next: boolean) => {
    window.localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
    window.dispatchEvent(new Event("storage"));
  }, []);
  return [collapsed, setCollapsed];
}

interface PaperSidebarProps {
  papers: PaperListItem[];
}

// Collapsible left navigation. On small screens it's a hamburger that opens a
// sheet; on large screens it's a persistent aside that collapses to a thin rail.
// The collapsed preference survives reloads via localStorage; it's read in an
// effect so the server and first client render agree.
export function PaperSidebar({ papers }: PaperSidebarProps) {
  const [collapsed, setCollapsed] = usePersistentCollapsed();
  const [sheetOpen, setSheetOpen] = useState(false);

  return (
    <>
      <header className="bg-background/80 sticky top-0 z-30 flex h-14 items-center gap-2 border-b px-3 backdrop-blur lg:hidden">
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          <SheetTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label="Open papers menu"
              />
            }
          >
            <Menu className="size-5" aria-hidden />
          </SheetTrigger>
          <SheetContent
            side="left"
            className="flex h-full w-80 flex-col gap-0 overflow-hidden p-0"
          >
            <SheetHeader className="border-b">
              <SheetTitle>Papers</SheetTitle>
            </SheetHeader>
            <SidebarList
              papers={papers}
              onNavigate={() => setSheetOpen(false)}
            />
          </SheetContent>
        </Sheet>
        <Link href="/" className="font-semibold">
          PhD for Dummies
        </Link>
      </header>

      <aside
        className={cn(
          "hidden shrink-0 border-r transition-[width] duration-200 lg:sticky lg:top-0 lg:flex lg:h-svh lg:flex-col",
          collapsed ? "lg:w-14" : "lg:w-72",
        )}
      >
        <div className="flex h-14 items-center justify-between gap-2 border-b px-3">
          {!collapsed && (
            <Link href="/" className="truncate font-semibold">
              PhD for Dummies
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!collapsed}
            className={cn(collapsed && "mx-auto")}
          >
            <PanelLeft className="size-4" aria-hidden />
          </Button>
        </div>
        {!collapsed && <SidebarList papers={papers} />}
      </aside>
    </>
  );
}
