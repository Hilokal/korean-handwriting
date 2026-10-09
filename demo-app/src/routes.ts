// Route table shared by the app (App.tsx) and the build (vite.config.ts
// prerenders one HTML file per route so link previews get per-page meta).
// Keep this file free of browser-only or `?raw` imports.

export type Lang = "en" | "ko";

export const SITE_URL = "https://hangul.ink";
export const OG_IMAGE = `${SITE_URL}/og.png`;
export const AUTHOR = "Jonathan Baudanza";

export interface RouteMeta {
  path: string;
  title: string; // <title> and og:title
  description: string;
  ogType: "website" | "article";
}

export interface PostMeta extends RouteMeta {
  ogType: "article";
  heading: string; // the h1 (title without the site suffix)
  date: string; // ISO
  dateLabel: string;
  blurb: string; // one line for the blog index
  ko: { heading: string; dateLabel: string; blurb: string };
}

export const HOME: RouteMeta = {
  path: "/",
  title: "hangul.ink — a neural network that writes Hangul by hand",
  description:
    "Type Korean text and a tiny recurrent network, trained on one writer's pen strokes, writes it out stroke by stroke — entirely in your browser.",
  ogType: "website",
};

export const BLOG_INDEX: RouteMeta = {
  path: "/blog",
  title: "Blog — hangul.ink",
  description: "Posts about building a neural network that writes Hangul by hand.",
  ogType: "website",
};

export const POSTS: PostMeta[] = [
  {
    path: "/blog/hangul-day",
    heading: "A neural network that writes Hangul by hand",
    title: "A neural network that writes Hangul by hand — hangul.ink",
    description:
      "Released for Hangul Day 2026: how I collected 4,500 lines of handwriting with a smartpen, adapted Alex Graves' architecture to Hangul, and published the dataset.",
    ogType: "article",
    date: "2026-10-09",
    dateLabel: "October 9, 2026",
    blurb:
      "Why I built it, how the data was collected, and what I'd try next.",
    ko: {
      heading: "손글씨로 한글을 쓰는 신경망",
      dateLabel: "2026년 10월 9일",
      blurb: "왜 만들었는지, 데이터를 어떻게 모았는지, 다음에 해 볼 만한 것들.",
    },
  },
  {
    path: "/blog/graves-departures",
    heading: "Departures from the Graves handwriting model",
    title: "Departures from the Graves handwriting model — hangul.ink",
    description:
      "Jamo embeddings, pen pressure, positional feedback, a trained stopping mechanism, and GRUs: where the Hangul model diverges from Graves (2013).",
    ogType: "article",
    date: "2026-10-09",
    dateLabel: "October 9, 2026",
    blurb: "The architecture changes it took to make Graves' model write Hangul.",
    ko: {
      heading: "Graves 손글씨 모델과 달라진 점",
      dateLabel: "2026년 10월 9일",
      blurb: "Graves의 모델이 한글을 쓰게 만들기까지 바꾼 아키텍처.",
    },
  },
];

export const ROUTES: RouteMeta[] = [HOME, BLOG_INDEX, ...POSTS];
