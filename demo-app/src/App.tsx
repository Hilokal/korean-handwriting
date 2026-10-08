import { useState } from "react";
import Demo from "./components/Demo";
import Markdown from "./components/Markdown";
import { LangToggle } from "./i18n";
import departuresMd from "./posts/graves-departures.md?raw";
import hangulDayMd from "./posts/hangul-day.md?raw";

const GITHUB_URL = "https://github.com/Hilokal/korean-handwriting";
const AUTHOR = "Jonathan Baudanza";

// The post is authored in markdown; `<!-- demo -->` marks where the live
// demo is embedded. Everything above it is the intro, everything below the
// write-up.
const DEMO_MARKER = "<!-- demo -->";

interface Post {
  path: string;
  title: string;
  date: string; // ISO, for <time>
  dateLabel: string;
  markdown: string;
}

const POSTS: Post[] = [
  {
    path: "/",
    title: "A neural network that writes Hangul by hand",
    date: "2026-10-09",
    dateLabel: "October 9, 2026",
    markdown: hangulDayMd,
  },
  {
    path: "/graves-departures",
    title: "Departures from the Graves handwriting model",
    date: "2026-10-09",
    dateLabel: "October 9, 2026",
    markdown: departuresMd,
  },
];

function currentPost(): Post {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  return POSTS.find((p) => p.path === path) ?? POSTS[0];
}

export default function App() {
  const post = currentPost();
  const [modelVersion, setModelVersion] = useState<string | null>(null);

  const markerAt = post.markdown.indexOf(DEMO_MARKER);
  const intro = markerAt >= 0 ? post.markdown.slice(0, markerAt) : post.markdown;
  const body = markerAt >= 0 ? post.markdown.slice(markerAt + DEMO_MARKER.length) : "";
  const hasDemo = markerAt >= 0;

  return (
    <div className="page">
      <header>
        <a className="brand" href="/">
          hangul<span className="brand-dot">.</span>ink
        </a>
        <nav>
          <LangToggle />
          <a
            className="link-button"
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
        </nav>
      </header>

      <main>
        <article>
          <h1 className="post-title">{post.title}</h1>
          <p className="byline">
            {AUTHOR} · <time dateTime={post.date}>{post.dateLabel}</time>
          </p>
          <Markdown source={intro} />
          {hasDemo && <Demo onModelLoaded={setModelVersion} />}
          {body && <Markdown source={body} />}
        </article>
      </main>

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
