// The shape the server hands the client sidebar. Deliberately minimal: the
// browser only needs what it takes to list and search papers, not the full
// metadata (headline, date, images stay server-side).
export interface PaperListItem {
  slug: string;
  title: string;
  abstract: string;
}
