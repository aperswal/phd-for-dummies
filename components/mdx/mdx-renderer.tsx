import type { MDXComponents } from "mdx/types";
import { MDXRemote } from "next-mdx-remote/rsc";
import rehypePrettyCode, {
  type Options as PrettyCodeOptions,
} from "rehype-pretty-code";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";

import { baseMdxComponents } from "@/components/mdx/mdx-components";

// One dark theme for code in both light and dark site modes. Highlighting runs
// here on the server via Shiki, so no highlighter JS reaches the browser.
const prettyCodeOptions: PrettyCodeOptions = {
  theme: "github-dark-default",
  keepBackground: true,
};

interface MdxRendererProps {
  source: string;
  components?: MDXComponents;
}

export function MdxRenderer({ source, components }: MdxRendererProps) {
  return (
    <MDXRemote
      source={source}
      components={{ ...baseMdxComponents, ...components }}
      options={{
        mdxOptions: {
          remarkPlugins: [remarkGfm],
          rehypePlugins: [rehypeSlug, [rehypePrettyCode, prettyCodeOptions]],
        },
      }}
    />
  );
}
