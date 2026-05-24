import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

// Single source of truth for config. Read values from here, never from
// process.env directly, so a missing or malformed variable fails loudly at
// startup instead of surfacing as a confusing runtime error deep in a page.
export const env = createEnv({
  server: {
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
  },
  client: {
    NEXT_PUBLIC_SITE_URL: z.string().url().default("http://localhost:3000"),
  },
  runtimeEnv: {
    NODE_ENV: process.env.NODE_ENV,
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL,
  },
  emptyStringAsUndefined: true,
});
