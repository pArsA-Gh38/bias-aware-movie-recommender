"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api/client";
import type {
  FeedbackAction,
  FeedbackRequest,
  RecommendationResponse,
} from "@/lib/api/types";

const ACTIONS: readonly FeedbackAction[] = ["click", "like", "dislike"];

type FeedbackFailure = "event" | "membership" | "validation" | "temporary" | "unexpected";
type RequestIdentity = {
  controller: AbortController;
  intent: FeedbackRequest;
  token: symbol;
};

function actionLabel(action: FeedbackAction): string {
  return action[0].toUpperCase() + action.slice(1);
}

function failureFor(error: unknown): FeedbackFailure {
  if (!(error instanceof ApiError)) return "unexpected";
  if (error.status === 404) return "event";
  if (error.status === 409) return "membership";
  if (error.status === 422) return "validation";
  if (error.status === 503 || error.status === 504) return "temporary";
  return "unexpected";
}

const FAILURE_MESSAGES: Record<FeedbackFailure, string> = {
  event: "This recommendation event is unavailable. Generate recommendations again.",
  membership: "This movie no longer belongs to the recommendation event. Generate recommendations again.",
  validation: "The feedback request was rejected. Generate recommendations again.",
  temporary: "Feedback is temporarily unavailable. Please try again.",
  unexpected: "Feedback could not be recorded. Please try again.",
};

function FeedbackControls({ eventId, movieId }: { eventId: string; movieId: number }) {
  const router = useRouter();
  const [phase, setPhase] = useState<"idle" | "submitting" | "failed" | "redirecting">("idle");
  const [confirmedAction, setConfirmedAction] = useState<FeedbackAction | null>(null);
  const [submittingAction, setSubmittingAction] = useState<FeedbackAction | null>(null);
  const [failedIntent, setFailedIntent] = useState<FeedbackRequest | null>(null);
  const [failure, setFailure] = useState<FeedbackFailure | null>(null);
  const requestRef = useRef<RequestIdentity | null>(null);
  const mountedRef = useRef(true);
  const errorRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<FeedbackAction | null>(null);
  const actionButtonRefs = useRef<Partial<Record<FeedbackAction, HTMLButtonElement | null>>>({});

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestRef.current?.controller.abort("Recommendation result was replaced");
    };
  }, []);

  useEffect(() => {
    if (phase === "failed") errorRef.current?.focus();
    if (phase === "idle" && restoreFocusRef.current) {
      actionButtonRefs.current[restoreFocusRef.current]?.focus();
      restoreFocusRef.current = null;
    }
  }, [phase]);

  async function submit(intent: FeedbackRequest, restoreFocus = false) {
    if (requestRef.current) return;
    const request: RequestIdentity = {
      controller: new AbortController(),
      intent,
      token: Symbol(`${intent.event_id}:${intent.movie_id}:${intent.action}`),
    };
    requestRef.current = request;
    setSubmittingAction(intent.action);
    setFailedIntent(null);
    setFailure(null);
    setPhase("submitting");

    try {
      await api.feedback(intent, request.controller.signal);
      if (!mountedRef.current || requestRef.current?.token !== request.token || request.controller.signal.aborted) return;
      setConfirmedAction(intent.action);
      setSubmittingAction(null);
      if (restoreFocus) restoreFocusRef.current = intent.action;
      setPhase("idle");
    } catch (caught) {
      if (!mountedRef.current || requestRef.current?.token !== request.token || request.controller.signal.aborted) return;
      if (caught instanceof ApiError && caught.status === 401) {
        setSubmittingAction(null);
        setPhase("redirecting");
        router.replace("/login");
        return;
      }
      setFailedIntent(intent);
      setSubmittingAction(null);
      setFailure(failureFor(caught));
      setPhase("failed");
    } finally {
      if (requestRef.current?.token === request.token) requestRef.current = null;
    }
  }

  const busy = phase === "submitting" || phase === "redirecting";
  return <fieldset className="recommendation-feedback" aria-busy={busy} disabled={busy}>
    <legend>Feedback for Movie ID {movieId}</legend>
    <div className="recommendation-feedback__actions">
      {ACTIONS.map((action) => <button
        key={action}
        ref={(element) => { actionButtonRefs.current[action] = element; }}
        type="button"
        aria-label={`Record ${actionLabel(action)} feedback for Movie ID ${movieId}`}
        onClick={() => void submit({ event_id: eventId, movie_id: movieId, action })}
      >
        {actionLabel(action)}
      </button>)}
    </div>
    {phase === "submitting" && <p className="recommendation-feedback__pending" role="status" aria-live="polite">
      Recording {actionLabel(submittingAction ?? "click")} feedback…
    </p>}
    {confirmedAction && <p className="recommendation-feedback__confirmation" role="status" aria-live="polite">
      Recorded {actionLabel(confirmedAction)} for Movie ID {movieId}.
    </p>}
    {phase === "failed" && failedIntent && failure && <div
      className="recommendation-feedback__error"
      role="alert"
      tabIndex={-1}
      ref={errorRef}
    >
      <p>{FAILURE_MESSAGES[failure]}</p>
      <button
        type="button"
        aria-label={`Retry ${actionLabel(failedIntent.action)} feedback for Movie ID ${movieId}`}
        onClick={() => void submit(failedIntent, true)}
      >
        Retry {actionLabel(failedIntent.action)}
      </button>
    </div>}
  </fieldset>;
}

export function RecommendationList({ result }: { result: RecommendationResponse }) {
  return <div className="recommendation-page__list">
    <dl className="recommendation-page__metadata" aria-label="Applied recommendation metadata">
      <div><dt>Configuration key</dt><dd>{result.configuration_key}</dd></div>
      <div><dt>Mode</dt><dd>{result.mode}</dd></div>
      <div><dt>Lambda</dt><dd>{result.lambda.toFixed(1)}</dd></div>
      <div><dt>Model version</dt><dd>{result.model_version_id}</dd></div>
      <div><dt>Strategy</dt><dd>{result.strategy}</dd></div>
      <div><dt>Event ID</dt><dd>{result.event_id}</dd></div>
    </dl>
    {result.items.length === 0
      ? <p className="recommendation-page__empty">No recommendations are available yet.</p>
      : <ol className="recommendation-page__recommendations" aria-label="Ranked movie recommendations">
        {result.items.map((item) => <li className="recommendation-page__row" key={`${result.event_id}:${item.movie_id}`}>
          <strong className="recommendation-page__rank">#{item.rank}</strong>
          <div className="recommendation-page__movie">
            <strong className="recommendation-page__title">{item.title}</strong>
            {item.genres.length > 0 && <span className="recommendation-page__genres">{item.genres.join(" · ")}</span>}
            <span className="recommendation-page__movie-id">Movie ID {item.movie_id}</span>
          </div>
          <span className="recommendation-page__score">Score {item.score.toFixed(6)}</span>
          <FeedbackControls eventId={result.event_id} movieId={item.movie_id} />
        </li>)}
      </ol>}
  </div>;
}
