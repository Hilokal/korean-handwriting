import { useState } from "react";
import Home from "./components/Home";
import Markdown from "./components/Markdown";
import Demo from "./components/Demo";
import { LangToggle, useI18n } from "./i18n";
import departuresKo from "./posts/graves-departures.ko.md?raw";
import departuresMd from "./posts/graves-departures.md?raw";
import hangulDayKo from "./posts/hangul-day.ko.md?raw";
import hangulDayMd from "./posts/hangul-day.md?raw";
import { AUTHOR, BLOG_INDEX, Lang, POSTS, PostMeta } from "./routes";

const GITHUB_URL = "https://github.com/Hilokal/korean-handwriting";
const DATASET_URL = "https://huggingface.co/datasets/jbaudanza/korean-handwriting-strokes";

// Posts are authored in markdown, one file per language; `<!-- demo -->`
// marks where the live demo is embedded (the article stays self-contained
// for readers who land on it). The language toggle switches the prose.
const DEMO_MARKER = "<!-- demo -->";
const MARKDOWN: Record<string, Record<Lang, string>> = {
  "/blog/hangul-day": { en: hangulDayMd, ko: hangulDayKo },
  "/blog/graves-departures": { en: departuresMd, ko: departuresKo },
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
  const { lang } = useI18n();
  const md = MARKDOWN[post.path]?.[lang] ?? MARKDOWN[post.path]?.en ?? "";
  const at = md.indexOf(DEMO_MARKER);
  const intro = at >= 0 ? md.slice(0, at) : md;
  const body = at >= 0 ? md.slice(at + DEMO_MARKER.length) : "";
  const heading = lang === "ko" ? post.ko.heading : post.heading;
  const dateLabel = lang === "ko" ? post.ko.dateLabel : post.dateLabel;
  return (
    <article lang={lang}>
      <h1 className="post-title">{heading}</h1>
      <p className="byline">
        {AUTHOR} · <time dateTime={post.date}>{dateLabel}</time>
      </p>
      <Markdown source={intro} />
      {at >= 0 && <Demo onModelLoaded={onModelLoaded} />}
      {body && <Markdown source={body} />}
    </article>
  );
}

function BlogIndex() {
  const { lang, s } = useI18n();
  return (
    <section className="blog-index">
      <h1 className="post-title">{s.navBlog}</h1>
      <ul>
        {POSTS.map((p) => (
          <li key={p.path}>
            <a href={p.path}>{lang === "ko" ? p.ko.heading : p.heading}</a>
            <span className="byline">
              <time dateTime={p.date}>
                {lang === "ko" ? p.ko.dateLabel : p.dateLabel}
              </time>
            </span>
            <p>{lang === "ko" ? p.ko.blurb : p.blurb}</p>
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
          <a
            className="link-button"
            href={DATASET_URL}
            target="_blank"
            rel="noreferrer"
          >
            Hugging Face ↗
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
