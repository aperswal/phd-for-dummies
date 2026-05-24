import type { MDXComponents } from "mdx/types";
import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MdxRenderer } from "@/components/mdx/mdx-renderer";
import { getVisualization } from "@/components/visualizations/registry";
import { getPaper } from "@/lib/papers/get-paper";
import { getAllPapers } from "@/lib/papers/get-papers";

interface PaperPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const papers = await getAllPapers();
  return papers.map((paper) => ({ slug: paper.slug }));
}

export async function generateMetadata({
  params,
}: PaperPageProps): Promise<Metadata> {
  const { slug } = await params;
  const paper = await getPaper(slug);
  if (!paper) return {};

  const { meta } = paper;
  const hero = meta.images[0];
  return {
    title: meta.title,
    description: meta.headline,
    openGraph: {
      title: meta.title,
      description: meta.headline,
      type: "article",
      images: hero ? [{ url: hero.src, alt: hero.alt }] : [],
    },
  };
}

export default async function PaperPage({ params }: PaperPageProps) {
  const { slug } = await params;
  const paper = await getPaper(slug);
  if (!paper) notFound();

  const { mdx } = paper;
  const Visualization = getVisualization(slug);
  const components: MDXComponents = {};
  if (Visualization) {
    components.Visualization = Visualization;
  }

  return (
    <article className="prose prose-zinc prose-code:before:content-none prose-code:after:content-none prose-pre:overflow-x-auto prose-pre:rounded-xl prose-pre:px-5 prose-pre:py-4 mx-auto w-full max-w-3xl px-4 py-10 sm:px-6 lg:py-14">
      <MdxRenderer source={mdx} components={components} />
    </article>
  );
}
