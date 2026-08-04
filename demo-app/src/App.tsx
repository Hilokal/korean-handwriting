import { useEffect, useRef, useState } from "react";
import FeedbackDialog from "./components/FeedbackDialog";
import HandwritingCanvas, { DotStore } from "./components/HandwritingCanvas";
import { LangToggle, useI18n } from "./i18n";
import { toAnimatedSvg } from "./lib/animatedSvg";
import { downloadBlob, slugify } from "./lib/download";
import { FeedbackPayload, sendFeedback } from "./lib/feedback";
import { HandwritingModel } from "./lib/generator";
import { randomSeed } from "./lib/prng";

const GITHUB_URL = "https://github.com/Hilokal/korean-handwriting";

const SAMPLES: { text: string; gloss: string }[] = [
  { text: "안녕하세요", gloss: "Hello" },
  { text: "만나서 반갑습니다.", gloss: "Nice to meet you." },
  { text: "오늘 날씨가 참 좋네요.", gloss: "The weather is lovely today." },
  { text: "한글은 아름다운 문자입니다.", gloss: "Hangul is a beautiful script." },
  { text: "커피 한 잔 어때요?", gloss: "How about a cup of coffee?" },
];

type Status = "loading" | "ready" | "generating" | "error";

export default function App() {
  const { s, lang } = useI18n();

  const modelRef = useRef<HandwritingModel | null>(null);
  const storeRef = useRef<DotStore>({ dots: [] });
  const abortRef = useRef<AbortController | null>(null);

  const [status, setStatus] = useState<Status>("loading");
  const [text, setText] = useState("안녕하세요");
  const [temperature, setTemperature] = useState(1.0);
  const [bias, setBias] = useState(0.5);
  const [seed, setSeed] = useState<number>(() => randomSeed());
  const [playNonce, setPlayNonce] = useState(0);
  const [hasResult, setHasResult] = useState(false);
  const [statusLine, setStatusLine] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);

  useEffect(() => {
    const t0 = performance.now();
    HandwritingModel.load()
      .then((m) => {
        modelRef.current = m;
        setStatus("ready");
        setStatusLine(s.loadedIn(performance.now() - t0));
      })
      .catch((err) => {
        console.error("model load failed", err);
        setStatus("error");
      });
    // load once; the status line is cosmetic, skip re-running on lang change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generate = async (genText: string, genSeed: number) => {
    const model = modelRef.current;
    if (!model || !genText.trim()) return;
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;

    storeRef.current.dots = [];
    setPlayNonce((n) => n + 1);
    setHasResult(false);
    setFeedbackSent(false);
    setStatus("generating");
    const t0 = performance.now();

    try {
      const gen = model.generate(
        { text: genText, temperature, bias, seed: genSeed },
        abort.signal,
      );
      for await (const dot of gen) {
        storeRef.current.dots.push(dot);
      }
      if (!abort.signal.aborted) {
        setStatus("ready");
        setHasResult(storeRef.current.dots.length > 0);
        setStatusLine(
          s.generatedIn(storeRef.current.dots.length, performance.now() - t0),
        );
      }
    } catch (err) {
      console.error("generation failed", err);
      if (!abort.signal.aborted) setStatus("ready");
    }
  };

  const onGenerate = () => {
    // A fresh seed per click, so repeat clicks explore the model's variety;
    // the seed box still shows exactly what produced the current strokes.
    const s2 = randomSeed();
    setSeed(s2);
    void generate(text, s2);
  };

  const onStop = () => {
    abortRef.current?.abort();
    setStatus("ready");
    setHasResult(storeRef.current.dots.length > 0);
  };

  const dotsJson = () =>
    JSON.stringify({ dots: storeRef.current.dots }, null, 2);

  const feedbackBase = (): Omit<FeedbackPayload, "rating" | "comment"> => ({
    text,
    temperature,
    bias,
    seed,
    modelVersion: modelRef.current?.meta.version ?? "unknown",
    dots: storeRef.current.dots,
    locale: lang,
  });

  const sendGood = async () => {
    setFeedbackSent(true); // optimistic; a lost 👍 is not worth an error state
    try {
      await sendFeedback({ ...feedbackBase(), rating: "good" });
    } catch (err) {
      console.warn("feedback failed", err);
    }
  };

  const generating = status === "generating";

  return (
    <div className="page">
      <header>
        <div className="brand">
          <h1>{s.title}</h1>
          <p className="tagline">{s.tagline}</p>
        </div>
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
        <section className="card input-card">
          <label htmlFor="text-input">{s.inputLabel}</label>
          <div className="input-row">
            <input
              id="text-input"
              type="text"
              value={text}
              maxLength={40}
              placeholder={s.inputPlaceholder}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !generating) onGenerate();
              }}
            />
            <button
              className="primary"
              disabled={status === "loading" || status === "error" || !text.trim()}
              onClick={generating ? onStop : onGenerate}
            >
              {generating ? s.stop : s.generate}
            </button>
          </div>

          <div className="samples">
            <span className="samples-label">{s.samples}:</span>
            {SAMPLES.map((sample) => (
              <button
                key={sample.text}
                className="chip"
                title={lang === "en" ? sample.gloss : undefined}
                onClick={() => {
                  setText(sample.text);
                  const s2 = randomSeed();
                  setSeed(s2);
                  void generate(sample.text, s2);
                }}
              >
                {sample.text}
                {lang === "en" && <span className="gloss">{sample.gloss}</span>}
              </button>
            ))}
          </div>

          <details className="params">
            <summary>{s.parameters}</summary>
            <div className="param">
              <label>
                {s.temperature}: <code>{temperature.toFixed(2)}</code>
              </label>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
              <p className="hint">{s.temperatureHint}</p>
            </div>
            <div className="param">
              <label>
                {s.bias}: <code>{bias.toFixed(1)}</code>
              </label>
              <input
                type="range"
                min={0}
                max={3}
                step={0.1}
                value={bias}
                onChange={(e) => setBias(Number(e.target.value))}
              />
              <p className="hint">{s.biasHint}</p>
            </div>
            <div className="param seed-param">
              <label>
                {s.seed}: <code>{seed}</code>
              </label>
              <button className="link-button" onClick={() => setSeed(randomSeed())}>
                {s.newSeed}
              </button>
              <p className="hint">{s.seedHint}</p>
            </div>
            <p className="hint">{s.onlyKorean}</p>
          </details>
        </section>

        <section className="card canvas-card">
          {status === "loading" && <p className="center-note">{s.modelLoading}</p>}
          {status === "error" && <p className="center-note error">{s.modelError}</p>}
          {(status === "ready" || generating) && (
            <>
              {storeRef.current.dots.length === 0 && !generating && (
                <p className="center-note">{s.emptyState}</p>
              )}
              <HandwritingCanvas store={storeRef.current} playNonce={playNonce} />
              <div className="canvas-footer">
                <span className="status-line">{statusLine}</span>
                {hasResult && (
                  <div className="actions">
                    <button
                      className="link-button"
                      onClick={() => setPlayNonce((n) => n + 1)}
                    >
                      ▶ {s.replay}
                    </button>
                    <button
                      className="link-button"
                      onClick={() =>
                        downloadBlob(
                          toAnimatedSvg(storeRef.current.dots),
                          `${slugify(text)}.svg`,
                          "image/svg+xml",
                        )
                      }
                    >
                      ⬇ {s.downloadSvg}
                    </button>
                    <button
                      className="link-button"
                      onClick={() =>
                        downloadBlob(
                          dotsJson(),
                          `${slugify(text)}.json`,
                          "application/json",
                        )
                      }
                    >
                      ⬇ {s.downloadJson}
                    </button>
                    {feedbackSent ? (
                      <span className="thanks">{s.feedbackThanks}</span>
                    ) : (
                      <>
                        <button className="link-button" onClick={sendGood}>
                          👍 {s.feedbackGood}
                        </button>
                        <button
                          className="link-button"
                          onClick={() => setShowFeedback(true)}
                        >
                          👎 {s.feedbackBad}
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </section>

        <p className="about">{s.about}</p>
      </main>

      <footer>
        <a href={GITHUB_URL} target="_blank" rel="noreferrer">
          {GITHUB_URL.replace("https://", "")}
        </a>
        <span>
          {s.footerModel}{" "}
          <code>{modelRef.current?.meta.version ?? "…"}</code>
        </span>
      </footer>

      {showFeedback && (
        <FeedbackDialog
          onClose={() => setShowFeedback(false)}
          onSubmit={async (comment) => {
            await sendFeedback({ ...feedbackBase(), rating: "bad", comment });
            setFeedbackSent(true);
          }}
        />
      )}
    </div>
  );
}
