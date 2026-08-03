import { createContext, ReactNode, useContext, useEffect, useState } from "react";

// Worker-facing strings only — the admin pages stay English. Default is
// Korean since the freelance workers are mostly Korean speakers.

const strings = {
  ko: {
    appTitle: "손글씨 수집",
    loading: "로딩 중…",
    logOut: "로그아웃",
    admin: "관리자",

    // Login
    username: "아이디",
    password: "비밀번호",
    logIn: "로그인",
    loggingIn: "로그인 중…",
    loginFailed: "아이디 또는 비밀번호가 올바르지 않습니다.",

    // Invite
    welcome: (name: string) => `${name}님, 환영합니다`,
    choosePassword: "계정 설정을 위해 비밀번호를 정해 주세요.",
    passwordMin: "비밀번호 (10자 이상)",
    confirmPassword: "비밀번호 확인",
    passwordMismatch: "비밀번호가 일치하지 않습니다.",
    createAccount: "계정 만들기",
    creatingAccount: "설정 중…",
    inviteNotFound: "초대장을 찾을 수 없습니다",
    inviteNotFoundBody:
      "이 초대 링크는 유효하지 않거나, 만료되었거나, 이미 사용되었습니다.",

    // Record
    connectPen: "펜 연결",
    disconnectPen: "펜 연결 해제",
    completedSession: (n: number) => `이번 세션에서 완료: ${n}`,
    coverage: (covered: number, total: number) =>
      `내 음절 커버리지: ${covered}/${total}`,
    penDisconnected:
      "펜 연결이 끊어졌습니다 — 계속하려면 다시 연결하거나, '다시 쓰기'를 눌러 주세요.",
    pageBoundary:
      "페이지 경계를 넘어 쓰신 것 같습니다. 한 문장은 한 페이지 안에 써 주세요. (필요하면 '다시 쓰기')",
    writePrompt: "아래 문장을 Ncode 종이에 써 주세요:",
    partIndicator: (i: number, n: number) => `부분 ${i} / ${n}`,
    connectToBegin: "시작하려면 펜을 연결해 주세요.",
    startWriting: "쓰기 시작하면 자동으로 기록이 시작됩니다.",
    recordingStatus: (strokes: number, dots: number) =>
      `기록 중… 획 ${strokes}개, 점 ${dots}개`,
    doneWriting: "다 썼어요",
    redo: "다시 쓰기",
    skipSentence: "문장 건너뛰기",
    reportProblem: "문제 신고",
    reportPlaceholder: "이 문장에 어떤 문제가 있나요?",
    sendReport: "신고 보내기",
    checkHandwriting: "손글씨를 확인해 주세요:",
    submit: "제출",
    submitting: "제출 중…",
    allDone: "모두 완료했습니다!",
    allDoneBody: "지금은 더 배정할 문장이 없습니다. 감사합니다!",
    browserNotSupported: "지원되지 않는 브라우저입니다",
    browserNotSupportedBody:
      "이 도구는 블루투스로 Neo 스마트펜에 연결합니다. 데스크톱 또는 Android의 Google Chrome / Microsoft Edge를 사용해 주세요. Safari와 iOS는 지원되지 않습니다.",
    errorGeneric: "오류가 발생했습니다. 다시 시도해 주세요.",
  },
  en: {
    appTitle: "Handwriting Collection",
    loading: "Loading…",
    logOut: "log out",
    admin: "admin",

    username: "Username",
    password: "Password",
    logIn: "Log in",
    loggingIn: "Logging in…",
    loginFailed: "Invalid username or password.",

    welcome: (name: string) => `Welcome, ${name}`,
    choosePassword: "Choose a password to finish setting up your account.",
    passwordMin: "Password (at least 10 characters)",
    confirmPassword: "Confirm password",
    passwordMismatch: "Passwords do not match.",
    createAccount: "Create account",
    creatingAccount: "Setting up…",
    inviteNotFound: "Invite not found",
    inviteNotFoundBody: "This invite link is invalid, expired, or already used.",

    connectPen: "Connect Pen",
    disconnectPen: "Disconnect Pen",
    completedSession: (n: number) => `Completed this session: ${n}`,
    coverage: (covered: number, total: number) =>
      `Your coverage: ${covered}/${total} syllables`,
    penDisconnected: "Pen disconnected — reconnect to continue, or Redo to start over.",
    pageBoundary:
      "Looks like you wrote across a page boundary. Please write each sentence on a single page (Redo if needed).",
    writePrompt: "Write this sentence on your Ncode paper:",
    partIndicator: (i: number, n: number) => `Part ${i} of ${n}`,
    connectToBegin: "Connect your pen to begin.",
    startWriting: "Start writing — recording begins automatically at the first pen stroke.",
    recordingStatus: (strokes: number, dots: number) =>
      `Recording… ${strokes} strokes, ${dots} points captured.`,
    doneWriting: "Done writing",
    redo: "Redo",
    skipSentence: "Skip sentence",
    reportProblem: "Report a problem",
    reportPlaceholder: "What's wrong with this sentence?",
    sendReport: "Send report",
    checkHandwriting: "Check your handwriting:",
    submit: "Submit",
    submitting: "Submitting…",
    allDone: "All done!",
    allDoneBody: "There are no more sentences available for you right now. Thank you!",
    browserNotSupported: "Browser not supported",
    browserNotSupportedBody:
      "This tool connects to your Neo Smartpen over Bluetooth, which requires Google Chrome or Microsoft Edge on desktop (or Android). Safari and iOS are not supported.",
    errorGeneric: "Something went wrong. Please try again.",
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

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const stored = localStorage.getItem("lang");
    return stored === "en" || stored === "ko" ? stored : "ko";
  });
  const setLang = (l: Lang) => {
    localStorage.setItem("lang", l);
    setLangState(l);
  };
  useEffect(() => {
    document.documentElement.lang = lang;
    document.title = strings[lang].appTitle;
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
