import { FormEvent, useState } from "react";
import { useI18n } from "../i18n";

export default function FeedbackDialog({
  onSubmit,
  onClose,
}: {
  onSubmit: (comment: string) => Promise<void>;
  onClose: () => void;
}) {
  const { s } = useI18n();
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSending(true);
    setError(false);
    try {
      await onSubmit(comment.trim());
      onClose();
    } catch {
      setError(true);
      setSending(false);
    }
  };

  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <form
        className="dialog"
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
      >
        <h3>{s.feedbackTitle}</h3>
        <textarea
          autoFocus
          rows={3}
          maxLength={500}
          placeholder={s.feedbackPlaceholder}
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        {error && <p className="error">{s.feedbackError}</p>}
        <div className="dialog-actions">
          <button type="button" className="link-button" onClick={onClose}>
            ✕
          </button>
          <button type="submit" className="primary" disabled={sending}>
            {sending ? s.feedbackSending : s.feedbackSend}
          </button>
        </div>
      </form>
    </div>
  );
}
