import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Assignment } from "../api";
import { RecordedDot } from "../pen/types";
import { usePenConnection } from "../pen/usePenConnection";
import { ConnectionButton } from "../pen/ConnectionButton";
import { analyzeDots, recordedDotsToSvgPath } from "../components/strokeSvg";
import { LiveInkCanvas } from "../components/LiveInkCanvas";
import { useI18n } from "../i18n";

// A handful of off-page move dots are Ncode misreads (filtered as noise);
// a large fraction means the worker genuinely switched pages mid-chunk.
const PAGE_CHANGE_MIN_DOTS = 20;
const PAGE_CHANGE_MIN_RATIO = 0.05;

type Phase = "writing" | "review";

const DRAFT_KEY = "recording-draft";
const DRAFT_SAVE_MS = 2000;

interface Draft {
  assignmentId: number;
  chunkIndex: number;
  dots: RecordedDot[];
}

function loadDraft(assignmentId: number, chunkIndex: number): RecordedDot[] | null {
  try {
    const raw = sessionStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const draft = JSON.parse(raw) as Draft;
    return draft.assignmentId === assignmentId &&
      draft.chunkIndex === chunkIndex &&
      draft.dots.length > 0
      ? draft.dots
      : null;
  } catch {
    return null;
  }
}

function saveDraft(assignmentId: number, chunkIndex: number, dots: RecordedDot[]): void {
  try {
    sessionStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({ assignmentId, chunkIndex, dots }),
    );
  } catch {
    // quota exceeded or unavailable — drafts are best-effort
  }
}

function clearDraft(): void {
  try {
    sessionStorage.removeItem(DRAFT_KEY);
  } catch {
    // ignore
  }
}

export function RecordPage() {
  const { s } = useI18n();
  const [assignment, setAssignment] = useState<Assignment | null>(null);
  const [noWork, setNoWork] = useState(false);
  const [phase, setPhase] = useState<Phase>("writing");
  const [dotCount, setDotCount] = useState(0);
  const [strokeCount, setStrokeCount] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [doneCount, setDoneCount] = useState(0);

  const dotsRef = useRef<RecordedDot[]>([]);
  const lastDraftSaveRef = useRef(0);
  const phaseRef = useRef<Phase>("writing");
  phaseRef.current = phase;
  const assignmentRef = useRef<Assignment | null>(null);
  assignmentRef.current = assignment;

  const ingestDot = useCallback((dot: RecordedDot) => {
    const dots = dotsRef.current;
    dots.push(dot);
    setDotCount(dots.length);
    if (dot.dotType === 0) setStrokeCount((n) => n + 1);
  }, []);

  // Cleaned view of the buffer: majority-page + glitch filtering, recomputed
  // as dots stream in (dotCount is the change signal for the mutable ref).
  const analysis = useMemo(
    () => analyzeDots(dotsRef.current),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dotCount],
  );
  const pageChanged =
    analysis.offPageCount > PAGE_CHANGE_MIN_DOTS &&
    analysis.offPageCount / analysis.moveCount > PAGE_CHANGE_MIN_RATIO;

  const handleDot = useCallback(
    (dot: RecordedDot) => {
      if (phaseRef.current !== "writing") return;
      if (dot.dotType === 3) return; // hover: not ink, don't buffer
      ingestDot(dot);
      const now = Date.now();
      if (assignmentRef.current && now - lastDraftSaveRef.current > DRAFT_SAVE_MS) {
        lastDraftSaveRef.current = now;
        saveDraft(
          assignmentRef.current.assignmentId,
          assignmentRef.current.nextChunkIndex,
          dotsRef.current,
        );
      }
    },
    [ingestDot],
  );

  const { controller, isConnected, penMac } = usePenConnection(handleDot);

  const [stats, setStats] = useState<{ syllablesCovered: number; totalSyllables: number } | null>(
    null,
  );
  const refreshStats = useCallback(() => {
    api
      .get<{ stats: { syllablesCovered: number; totalSyllables: number } }>("/api/me")
      .then((r) => setStats(r.stats))
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshStats();
    api
      .get<Assignment>("/api/work/next")
      .then((a) => {
        setAssignment(a);
        // Restore an in-progress draft after a reload: the lease is
        // server-side, so the same assignment/chunk comes back.
        const draft = loadDraft(a.assignmentId, a.nextChunkIndex);
        if (draft) {
          for (const dot of draft) ingestDot(dot);
        }
      })
      .catch((e) => {
        if (e.status === 404) setNoWork(true);
        else setError(e.message);
      });
  }, [ingestDot, refreshStats]);

  const resetBuffer = useCallback(() => {
    dotsRef.current = [];
    clearDraft();
    setDotCount(0);
    setStrokeCount(0);
    setPhase("writing");
  }, []);

  const advanceTo = useCallback(
    (next: Assignment | null) => {
      resetBuffer();
      if (next) {
        setAssignment(next);
      } else {
        setAssignment(null);
        setNoWork(true);
      }
    },
    [resetBuffer],
  );

  const handleSubmit = useCallback(async () => {
    if (!assignment) return;
    const dots = dotsRef.current;
    setSubmitting(true);
    setError(null);
    try {
      const result = await api.post<{ recordingId: number; next: Assignment | null }>(
        `/api/work/${assignment.assignmentId}/submit`,
        {
          startTime: dots[0].timeStamp,
          endTime: dots[dots.length - 1].timeStamp,
          dotCount: dots.length,
          chunkIndex: assignment.nextChunkIndex,
          dots,
          penMac,
          pageInfo: analysis.modalPage ?? dots[0].pageInfo,
        },
      );
      setDoneCount((n) => n + 1);
      refreshStats();
      advanceTo(result.next);
    } catch (e) {
      setError(e instanceof Error ? e.message : s.errorGeneric);
    } finally {
      setSubmitting(false);
    }
  }, [assignment, penMac, advanceTo, refreshStats, analysis]);

  const handleSkip = useCallback(async () => {
    if (!assignment) return;
    setError(null);
    try {
      const result = await api.post<{ next: Assignment | null }>(
        `/api/work/${assignment.assignmentId}/skip`,
      );
      advanceTo(result.next);
    } catch (e) {
      setError(e instanceof Error ? e.message : s.errorGeneric);
    }
  }, [assignment, advanceTo]);

  const [reporting, setReporting] = useState(false);
  const [reportNote, setReportNote] = useState("");

  const handleReport = useCallback(async () => {
    if (!assignment) return;
    setError(null);
    try {
      const result = await api.post<{ next: Assignment | null }>(
        `/api/work/${assignment.assignmentId}/report`,
        { note: reportNote },
      );
      setReporting(false);
      setReportNote("");
      advanceTo(result.next);
    } catch (e) {
      setError(e instanceof Error ? e.message : s.errorGeneric);
    }
  }, [assignment, reportNote, advanceTo]);

  if (typeof navigator !== "undefined" && !("bluetooth" in navigator)) {
    return (
      <div className="page-center">
        <div className="card">
          <h1>{s.browserNotSupported}</h1>
          <p>{s.browserNotSupportedBody}</p>
        </div>
      </div>
    );
  }

  if (noWork) {
    return (
      <div className="page-center">
        <div className="card">
          <h1>{s.allDone}</h1>
          <p>{s.allDoneBody}</p>
        </div>
      </div>
    );
  }
  if (!assignment) {
    return <div className="page-center">{error ?? s.loading}</div>;
  }

  const svg =
    phase === "review" ? recordedDotsToSvgPath(dotsRef.current, 1 / 200) : null;

  return (
    <div className="record-page">
      <div className="record-status">
        <ConnectionButton isConnected={isConnected} controller={controller} />
        <span className="done-count">{s.completedSession(doneCount)}</span>
        {stats && (
          <span className="done-count">
            {s.coverage(stats.syllablesCovered, stats.totalSyllables)}
          </span>
        )}
      </div>

      {!isConnected && dotCount > 0 && (
        <p className="banner banner-warn">{s.penDisconnected}</p>
      )}
      {pageChanged && <p className="banner banner-warn">{s.pageBoundary}</p>}
      {error && <p className="banner banner-error">{error}</p>}

      <div className="card sentence-card">
        <p className="sentence-label">
          {s.writePrompt}
          {assignment.chunks.length > 1 && (
            <>
              {" · "}
              <strong>
                {s.partIndicator(
                  assignment.nextChunkIndex + 1,
                  assignment.chunks.length,
                )}
              </strong>
            </>
          )}
        </p>
        <p className="sentence-text">
          {assignment.chunks[assignment.nextChunkIndex] ?? assignment.text}
        </p>
        {assignment.chunks.length > 1 && (
          <p className="sentence-context">{assignment.text}</p>
        )}
      </div>

      {phase === "writing" && (
        <>
          <p className="hint">
            {!isConnected
              ? s.connectToBegin
              : dotCount === 0
                ? s.startWriting
                : s.recordingStatus(strokeCount, dotCount)}
          </p>
          {dotCount > 0 && <LiveInkCanvas points={analysis.points} />}
          <div className="actions">
            <button
              className="btn btn-primary"
              disabled={dotCount === 0}
              onClick={() => setPhase("review")}
            >
              {s.doneWriting}
            </button>
            {dotCount > 0 && (
              <button className="btn btn-secondary" onClick={resetBuffer}>
                {s.redo}
              </button>
            )}
            <button className="btn btn-tertiary" disabled={dotCount > 0} onClick={handleSkip}>
              {s.skipSentence}
            </button>
            <button
              className="btn btn-tertiary"
              disabled={dotCount > 0}
              onClick={() => setReporting(!reporting)}
            >
              {s.reportProblem}
            </button>
          </div>
          {reporting && (
            <div className="report-form">
              <input
                placeholder={s.reportPlaceholder}
                value={reportNote}
                onChange={(e) => setReportNote(e.target.value)}
              />
              <button className="btn btn-secondary" onClick={handleReport}>
                {s.sendReport}
              </button>
            </div>
          )}
        </>
      )}

      {phase === "review" && svg && (
        <>
          <div className="card review-card">
            <p className="sentence-label">{s.checkHandwriting}</p>
            <svg
              className="review-svg"
              viewBox={svg.viewBox}
              preserveAspectRatio="xMidYMid meet"
            >
              <path d={svg.pathData} fill="#222" stroke="none" />
            </svg>
          </div>
          <div className="actions">
            <button className="btn btn-primary" disabled={submitting} onClick={handleSubmit}>
              {submitting ? s.submitting : s.submit}
            </button>
            <button
              className="btn btn-secondary"
              disabled={submitting}
              onClick={resetBuffer}
            >
              {s.redo}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
