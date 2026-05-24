import { describe, expect, it } from "vitest";

import { paperMetaSchema } from "@/lib/papers/paper-schema";

const valid = {
  slug: "example-paper",
  title: "Example Paper",
  headline: "A one-line summary of what the paper shows.",
  abstract: "A short abstract used to exercise the metadata schema.",
  date: "2020-01-01",
  tags: ["example"],
  images: [{ src: "/papers/x/hero.png", alt: "A diagram of the mechanism" }],
};

describe("paperMetaSchema", () => {
  it("accepts well-formed metadata", () => {
    expect(paperMetaSchema.parse(valid)).toMatchObject({ slug: valid.slug });
  });

  it("rejects empty alt text", () => {
    const bad = { ...valid, images: [{ src: "/x.png", alt: "" }] };
    expect(paperMetaSchema.safeParse(bad).success).toBe(false);
  });

  it("rejects a non-kebab slug", () => {
    expect(
      paperMetaSchema.safeParse({ ...valid, slug: "Not Kebab" }).success,
    ).toBe(false);
  });

  it("rejects a malformed date", () => {
    expect(
      paperMetaSchema.safeParse({ ...valid, date: "June 2017" }).success,
    ).toBe(false);
  });

  it("defaults tags and images to empty arrays", () => {
    const parsed = paperMetaSchema.parse({
      slug: "x",
      title: "X",
      headline: "h",
      abstract: "a",
      date: "2020-01-01",
    });
    expect(parsed.tags).toEqual([]);
    expect(parsed.images).toEqual([]);
    expect(parsed.hasVisualization).toBe(false);
  });
});
