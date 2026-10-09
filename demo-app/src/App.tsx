import { useState } from "react";
import Home from "./components/Home";
import Markdown from "./components/Markdown";
import Demo from "./components/Demo";
import { LangToggle, useI18n } from "./i18n";
import departuresMd from "./posts/graves-departures.md?raw";
import hangulDayMd from "./posts/hangul-day.md?raw";
import { AUTHOR, BLOG_INDEX, POSTS, PostMeta } from "./routes";

const GITHUB_URL = "https://github.com/Hilokal/korean-handwriting";

// Posts are authored in markdown; `<!-- demo -->` marks where the live demo
// is embedded (the article stays self-contained for readers who land on it).
const DEMO_MARKER = "<!-- demo -->";
const MARKDOWN: Record<string, string> = {
  "/blog/hangul-day": hangulDayMd,
  "/blog/graves-departures": departuresMd,
};

function currentPath(): string {
  return window.location.pathname.replace(/\/+$/, "") || "/";
}

function Post({
  post,
  onModelLoaded,
}: {
  post: PostMeta;
  onModelLoaded: (v: string) => void;
}) {
  const md = MARKDOWN[post.path] ?? "";
  const at = md.indexOf(DEMO_MARKER);
  const intro = at >= 0 ? md.slice(0, at) : md;
  const body = at >= 0 ? md.slice(at + DEMO_MARKER.length) : "";
  return (
    <article>
      <h1 className="post-title">{post.heading}</h1>
      <p className="byline">
        {AUTHOR} · <time dateTime={post.date}>{post.dateLabel}</time>
      </p>
      <Markdown source={intro} />
      {at >= 0 && <Demo onModelLoaded={onModelLoaded} />}
      {body && <Markdown source={body} />}
    </article>
  );
}

function BlogIndex() {
  return (
    <section className="blog-index">
      <h1 className="post-title">Blog</h1>
      <ul>
        {POSTS.map((p) => (
          <li key={p.path}>
            <a href={p.path}>{p.heading}</a>
            <span className="byline">
              <time dateTime={p.date}>{p.dateLabel}</time>
            </span>
            <p>{p.blurb}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function App() {
  const { s } = useI18n();
  const path = currentPath();
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const post = POSTS.find((p) => p.path === path);

  let page;
  if (post) page = <Post post={post} onModelLoaded={setModelVersion} />;
  else if (path === BLOG_INDEX.path) page = <BlogIndex />;
  else page = <Home onModelLoaded={setModelVersion} />;

  return (
    <div className="page">
      <header>
        <a className="brand" href="/">
          hangul<span className="brand-dot">.</span>ink
        </a>
        <nav>
          <a className="link-button" href={BLOG_INDEX.path}>
            {s.navBlog}
          </a>
          <a
            className="link-button"
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
          <LangToggle />
        </nav>
      </header>

      <main>{page}</main>

      <footer>
        <a href={GITHUB_URL} target="_blank" rel="noreferrer">
          {GITHUB_URL.replace("https://", "")}
        </a>
        {modelVersion && (
          <span>
            model <code>{modelVersion}</code>
          </span>
        )}
      </footer>
    </div>
  );
}
