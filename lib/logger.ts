import "server-only";

import pino from "pino";

import { env } from "@/lib/env";

// One shared structured logger. Emits JSON lines so logs stay searchable in any
// aggregator. Server-only: it touches Node internals and must never ship to the
// browser bundle.
export const logger = pino({
  level: env.NODE_ENV === "production" ? "info" : "debug",
  base: { app: "research-paper-showcase" },
});
