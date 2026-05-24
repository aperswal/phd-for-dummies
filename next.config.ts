import type { NextConfig } from "next";

// MDX is compiled at request/build time by next-mdx-remote/rsc, not by Next's
// own loader, so the experimental Rust MDX compiler stays off (it would ignore
// our rehype plugins). pino ships its own transports and is kept external so
// the server bundler doesn't try to inline its worker code.
const nextConfig: NextConfig = {
  serverExternalPackages: ["pino"],
};

export default nextConfig;
