import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";
import { PaperSidebar } from "@/components/layout/paper-sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { env } from "@/lib/env";
import { getPaperListItems } from "@/lib/papers/get-papers";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(env.NEXT_PUBLIC_SITE_URL),
  title: {
    default: "PhD for Dummies",
    template: "%s — PhD for Dummies",
  },
  description:
    "Famous research papers, explained in layers. Start at the five-year-old version and climb as far as you want, with clean diagrams and a live demo you can play with.",
};

export const viewport: Viewport = {
  themeColor: "#c2683f",
};

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const papers = await getPaperListItems();
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      {/* Browser extensions (Grammarly, etc.) inject attributes like
          data-gr-ext-installed onto <body> before React hydrates, which would
          otherwise trip a hydration mismatch warning. */}
      <body className="min-h-svh" suppressHydrationWarning>
        <TooltipProvider delay={200}>
          <div className="flex min-h-svh flex-col lg:flex-row">
            <PaperSidebar papers={papers} />
            <main className="min-w-0 flex-1">{children}</main>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
