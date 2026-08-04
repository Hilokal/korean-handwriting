import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";

// Same pattern as collection-app/client/src/i18n.tsx. Locale is auto-detected
// from the browser on first visit, then remembered.

const strings = {
  en: {
    title: "hangul.ink",
    tagline: "A neural network writes Korean by hand — live in your browser.",
    about:
      "A recurrent network trained on real pen recordings draws each stroke, one point at a time. Nothing is sent to a server — the model runs right here.",
    inputLabel: "Korean text",
    inputPlaceholder: "Type Korean text… (or try a sample below)",
    generate: "Write it",
    generating: "Writing…",
    stop: "Stop",
    replay: "Replay",
    samples: "Samples",
    parameters: "Parameters",
    temperature: "Temperature",
    temperatureHint: "Pen-lift randomness: 0 = deterministic stroke breaks.",
    bias: "Bias",
    biasHint: "Higher = cleaner, more careful strokes. 0 = the model's natural hand.",
    seed: "Seed",
    newSeed: "New seed",
    seedHint: "Same seed + settings reproduces the same strokes.",
    downloadSvg: "Animated SVG",
    downloadJson: "Stroke JSON",
    feedbackGood: "Looks good",
    feedbackBad: "Looks wrong",
    feedbackThanks: "Thanks — this helps improve the model!",
    feedbackTitle: "What looks wrong?",
    feedbackPlaceholder:
      "Optional: which character or stroke came out wrong?",
    feedbackSend: "Send feedback",
    feedbackSending: "Sending…",
    feedbackError: "Could not send feedback. Please try again.",
    modelLoading: "Loading model…",
    modelError: "Could not load the model. Try refreshing the page.",
    onlyKorean:
      "Tip: the model was trained on Hangul, spaces, and . , ! ? — other characters are unknown to it.",
    emptyState: "The pen is ready.",
    footerModel: "model",
    loadedIn: (ms: number) => `model loaded in ${Math.round(ms)} ms`,
    generatedIn: (points: number, ms: number) =>
      `${points} points in ${Math.round(ms)} ms`,
  },
  ko: {
    title: "hangul.ink",
    tagline: "신경망이 브라우저에서 실시간으로 한글 손글씨를 씁니다.",
    about:
      "실제 펜 기록으로 학습된 순환 신경망이 한 점씩 획을 그립니다. 서버로 전송되는 것 없이 모델이 바로 여기서 실행됩니다.",
    inputLabel: "한글 텍스트",
    inputPlaceholder: "한글을 입력하세요… (아래 예문도 있어요)",
    generate: "쓰기",
    generating: "쓰는 중…",
    stop: "중지",
    replay: "다시 보기",
    samples: "예문",
    parameters: "설정",
    temperature: "온도",
    temperatureHint: "획 끊기의 무작위성: 0 = 항상 같은 위치에서 획을 뗍니다.",
    bias: "바이어스",
    biasHint: "높을수록 또박또박 씁니다. 0 = 모델의 자연스러운 필체.",
    seed: "시드",
    newSeed: "새 시드",
    seedHint: "같은 시드와 설정이면 같은 글씨가 재현됩니다.",
    downloadSvg: "애니메이션 SVG",
    downloadJson: "획 JSON",
    feedbackGood: "잘 나왔어요",
    feedbackBad: "이상해요",
    feedbackThanks: "감사합니다 — 모델 개선에 큰 도움이 됩니다!",
    feedbackTitle: "어떤 점이 이상한가요?",
    feedbackPlaceholder: "선택 사항: 어떤 글자나 획이 잘못 나왔나요?",
    feedbackSend: "보내기",
    feedbackSending: "보내는 중…",
    feedbackError: "전송에 실패했습니다. 다시 시도해 주세요.",
    modelLoading: "모델 로딩 중…",
    modelError: "모델을 불러오지 못했습니다. 새로고침해 주세요.",
    onlyKorean:
      "참고: 모델은 한글과 공백, . , ! ? 만 학습했습니다 — 그 외 문자는 알지 못합니다.",
    emptyState: "펜이 준비됐습니다.",
    footerModel: "모델",
    loadedIn: (ms: number) => `모델 로딩 ${Math.round(ms)} ms`,
    generatedIn: (points: number, ms: number) =>
      `${points}개 점, ${Math.round(ms)} ms`,
  },
} as const;

export type Lang = keyof typeof strings;
export type Strings = (typeof strings)[Lang];

interface I18n {
  lang: Lang;
  setLang: (lang: Lang) => void;
  s: Strings;
}

const I18nContext = createContext<I18n | null>(null);

function detectLang(): Lang {
  const stored = localStorage.getItem("lang");
  if (stored === "en" || stored === "ko") return stored;
  return navigator.language.toLowerCase().startsWith("ko") ? "ko" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectLang);
  const setLang = (l: Lang) => {
    localStorage.setItem("lang", l);
    setLangState(l);
  };
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
  return (
    <I18nContext.Provider value={{ lang, setLang, s: strings[lang] }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n(): I18n {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n outside I18nProvider");
  return ctx;
}

export function LangToggle() {
  const { lang, setLang } = useI18n();
  return (
    <button
      className="link-button"
      onClick={() => setLang(lang === "ko" ? "en" : "ko")}
    >
      {lang === "ko" ? "English" : "한국어"}
    </button>
  );
}
