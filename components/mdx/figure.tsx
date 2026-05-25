import type { ImgHTMLAttributes } from "react";

// Markdown images carry only src and alt, so we render a plain lazy <img>
// rather than next/image, which needs intrinsic dimensions we don't have here.
// Alt text is required upstream in meta.json; the fallback keeps a missing alt
// from ever reaching the DOM as undefined.
export function MdxImage({ src, alt }: ImgHTMLAttributes<HTMLImageElement>) {
  if (typeof src !== "string") return null;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- markdown images lack the intrinsic dimensions next/image requires
    <img
      src={src}
      alt={alt ?? ""}
      loading="lazy"
      className="mx-auto my-8 w-full rounded-xl"
    />
  );
}
