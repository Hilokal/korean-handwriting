import { Dot } from "./generator";
import { Lang } from "../i18n";

// The feedback endpoint lives in the collection app (same Fly app that stores
// the training data), so bad generations land next to the review tooling.
const API =
  import.meta.env.VITE_FEEDBACK_API ??
  "https://handwriting-collection.fly.dev/api/feedback";

export interface FeedbackPayload {
  rating: "good" | "bad";
  text: string;
  temperature: number;
  bias: number;
  seed: number;
  modelVersion: string;
  dots: Dot[];
  comment?: string;
  locale: Lang;
}

export async function sendFeedback(payload: FeedbackPayload): Promise<void> {
  const res = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`feedback failed: ${res.status}`);
}
