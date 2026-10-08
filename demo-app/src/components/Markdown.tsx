import { marked } from "marked";
import { useMemo } from "react";

// Renders one of the authored posts in src/posts/*.md. The markdown is our
// own content bundled at build time (vite `?raw`), so no sanitizer is needed.

marked.use({ gfm: true, breaks: false });

export default function Markdown({ source }: { source: string }) {
  const html = useMemo(() => marked.parse(source, { async: false }), [source]);
  return <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />;
}
