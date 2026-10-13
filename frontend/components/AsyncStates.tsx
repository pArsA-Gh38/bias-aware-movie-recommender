export function LoadingState({ label = "Loading" }: { label?: string }) {
  return <div className="state" role="status" aria-live="polite">{label}…</div>;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state" role="alert">
      <div className="stack">
        <p>{message}</p>
        {onRetry && <button className="button" onClick={onRetry}>Try again</button>}
      </div>
    </div>
  );
}

export function EmptyState({ message, action }: { message: string; action?: React.ReactNode }) {
  return <div className="state"><div><p>{message}</p>{action}</div></div>;
}
