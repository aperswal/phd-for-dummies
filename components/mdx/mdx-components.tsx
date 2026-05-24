import type { MDXComponents } from "mdx/types";

import { MdxImage } from "@/components/mdx/figure";

// Overrides applied to every paper's MDX. Per-paper components (the
// <Visualization /> for one slug) get merged on top of these by the renderer.
export const baseMdxComponents: MDXComponents = {
  img: MdxImage,
};
