// Provider-agnostic event tracking. Call track() everywhere; swap the body for
// a real sink (PostHog, GA, Plausible) in one place when one gets wired up.
// Safe to call from client components: no Node-only dependencies.

export type AnalyticsEvent =
  | { name: "paper_view"; slug: string }
  | { name: "visualization_play"; slug: string };

export function track(event: AnalyticsEvent): void {
  if (process.env.NODE_ENV !== "production") {
    console.debug("[analytics]", event.name, event);
  }
}
