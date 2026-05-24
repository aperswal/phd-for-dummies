import type { MetadataRoute } from "next";

// Drives the PWA install metadata and the maskable icons Android uses on the
// home screen. Next links this automatically as /manifest.webmanifest.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "PhD for Dummies",
    short_name: "PhD for Dummies",
    description:
      "Famous research papers, explained in layers from a five-year-old version up to a peer researcher, with diagrams and interactive demos.",
    start_url: "/",
    display: "standalone",
    background_color: "#f4ecd8",
    theme_color: "#c2683f",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      {
        src: "/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
