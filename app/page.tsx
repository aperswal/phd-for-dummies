import { PaperGrid } from "@/components/layout/paper-grid";
import { Card } from "@/components/ui/card";
import { getAllPapers } from "@/lib/papers/get-papers";

export default async function Home() {
  const papers = await getAllPapers();
  const cards = papers.map((paper) => ({
    slug: paper.slug,
    title: paper.title,
    headline: paper.headline,
  }));

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-10 sm:px-6 lg:px-10 lg:py-14">
      <header className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight">
          PhD for Dummies
        </h1>
        <p className="text-muted-foreground mt-2 max-w-2xl">
          Famous research papers, explained in layers. Start at the
          five-year-old version and climb as far as you want, up to the one a
          researcher would pick apart. Each comes with clean diagrams and a live
          demo you can play with.
        </p>
      </header>

      {papers.length === 0 ? (
        <Card className="p-8">
          <p className="text-muted-foreground">
            No papers yet. Drop a PDF into{" "}
            <code className="font-mono">papers/</code> and run{" "}
            <code className="font-mono">/add-paper</code> to add the first one.
          </p>
        </Card>
      ) : (
        <PaperGrid papers={cards} />
      )}
    </div>
  );
}
