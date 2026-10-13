import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { RecommendationList } from "@/components/RecommendationList";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const result = {
  event_id: "7c638ee0-21ca-4c50-a2ac-87f8d7232a89",
  model_version_id: "model-1",
  strategy: "als_fold_in",
  configuration_key: "A1_L07" as const,
  mode: "diversity_only" as const,
  lambda: 0.7,
  items: [{ movie_id: 99, title: "Authoritative Movie 99", genres: ["Drama", "Synthetic"], rank: 1, score: 0.81234567 }],
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return { promise, resolve };
}

function feedbackBody(callIndex = 0) {
  return JSON.parse(String(vi.mocked(fetch).mock.calls[callIndex]?.[1]?.body));
}

beforeEach(() => {
  replace.mockReset();
  vi.stubGlobal("fetch", vi.fn());
});

it("renders server metadata and only the available ranked result fields", () => {
  render(<RecommendationList result={result} />);
  expect(screen.getByText("A1_L07")).toBeInTheDocument();
  expect(screen.getByText("diversity_only")).toBeInTheDocument();
  expect(screen.getByText("0.7")).toBeInTheDocument();
  expect(screen.getByText(result.event_id)).toBeInTheDocument();
  expect(screen.getByText("model-1")).toBeInTheDocument();
  expect(screen.getByText("als_fold_in")).toBeInTheDocument();
  expect(screen.getByText("#1")).toBeInTheDocument();
  expect(screen.getByText("Movie ID 99")).toBeInTheDocument();
  expect(screen.getByText("Authoritative Movie 99")).toBeInTheDocument();
  expect(screen.getByText("Drama · Synthetic")).toBeInTheDocument();
  expect(screen.getByText("Score 0.812346")).toBeInTheDocument();
  expect(screen.queryByText(/genre|because|explanation|poster|popularity/i)).not.toBeInTheDocument();
});

it("retains all applied metadata when the successful result is empty", () => {
  render(<RecommendationList result={{ ...result, items: [] }} />);
  expect(screen.getByText("No recommendations are available yet.")).toBeInTheDocument();
  expect(screen.getByText("A1_L07")).toBeInTheDocument();
  expect(screen.getByText(result.event_id)).toBeInTheDocument();
  expect(screen.queryByRole("list", { name: "Ranked movie recommendations" })).not.toBeInTheDocument();
  expect(screen.queryByRole("group", { name: /Feedback for/ })).not.toBeInTheDocument();
});

it("renders three semantic actions with truthful item identity and makes no initial request", () => {
  render(<RecommendationList result={result} />);
  expect(screen.getByRole("group", { name: "Feedback for Movie ID 99" })).toBeInTheDocument();
  for (const action of ["Click", "Like", "Dislike"]) {
    expect(screen.getByRole("button", { name: `Record ${action} feedback for Movie ID 99` })).toBeEnabled();
  }
  expect(fetch).not.toHaveBeenCalled();
});

it("submits the exact visible identity and confirms only after the recorded response", async () => {
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValue(pending.promise);
  render(<RecommendationList result={result} />);

  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  expect(feedbackBody()).toEqual({ event_id: result.event_id, movie_id: 99, action: "like" });
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();

  pending.resolve(Response.json({ status: "recorded" }, { status: 201 }));
  expect(await screen.findByText("Recorded Like for Movie ID 99.")).toHaveAttribute("role", "status");
  expect(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" })).toBeEnabled();
});

it("locks only one item synchronously while distinct rows submit concurrently", async () => {
  const first = deferred<Response>();
  const second = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  render(<RecommendationList result={{ ...result, items: [
    result.items[0],
    { movie_id: 100, title: "Authoritative Movie 100", genres: ["Drama", "Synthetic"], rank: 2, score: 0.7 },
  ] }} />);

  const like99 = screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" });
  fireEvent.click(like99);
  fireEvent.click(like99);
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(like99).toBeDisabled();
  expect(screen.getByRole("group", { name: "Feedback for Movie ID 99" })).toHaveAttribute("aria-busy", "true");
  expect(screen.getByRole("button", { name: "Record Dislike feedback for Movie ID 99" })).toBeDisabled();

  await userEvent.click(screen.getByRole("button", { name: "Record Click feedback for Movie ID 100" }));
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("button", { name: "Record Like feedback for Movie ID 100" })).toBeDisabled();
  expect(feedbackBody(1)).toEqual({ event_id: result.event_id, movie_id: 100, action: "click" });

  first.resolve(Response.json({ status: "recorded" }, { status: 201 }));
  second.resolve(Response.json({ status: "recorded" }, { status: 201 }));
  expect(await screen.findByText("Recorded Like for Movie ID 99.")).toBeInTheDocument();
  expect(await screen.findByText("Recorded Click for Movie ID 100.")).toBeInTheDocument();
});

it("allows repeated append-only actions and preserves confirmation when a later action fails", async () => {
  vi.mocked(fetch)
    .mockResolvedValueOnce(Response.json({ status: "recorded" }, { status: 201 }))
    .mockResolvedValueOnce(Response.json({ detail: "private" }, { status: 503 }));
  render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  await screen.findByText("Recorded Like for Movie ID 99.");
  await userEvent.click(screen.getByRole("button", { name: "Record Dislike feedback for Movie ID 99" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Feedback is temporarily unavailable. Please try again.");
  expect(screen.getByText("Recorded Like for Movie ID 99.")).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledTimes(2);
});

it("retries the exact immutable failed intent and restores focus to its action after success", async () => {
  vi.mocked(fetch)
    .mockResolvedValueOnce(Response.json({ detail: "unavailable" }, { status: 503 }))
    .mockResolvedValueOnce(Response.json({ status: "recorded" }, { status: 201 }));
  render(<RecommendationList result={result} />);
  const action = screen.getByRole("button", { name: "Record Dislike feedback for Movie ID 99" });
  await userEvent.click(action);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveFocus();

  await userEvent.click(screen.getByRole("button", { name: "Retry Dislike feedback for Movie ID 99" }));
  await screen.findByText("Recorded Dislike for Movie ID 99.");
  expect(feedbackBody(0)).toEqual(feedbackBody(1));
  expect(action).toHaveFocus();
});

it.each([
  [404, "This recommendation event is unavailable. Generate recommendations again."],
  [409, "This movie no longer belongs to the recommendation event. Generate recommendations again."],
  [422, "The feedback request was rejected. Generate recommendations again."],
  [503, "Feedback is temporarily unavailable. Please try again."],
  [504, "Feedback is temporarily unavailable. Please try again."],
  [500, "Feedback could not be recorded. Please try again."],
] as const)("shows a sanitized focused alert for status %s", async (status, message) => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "sensitive upstream" }, { status }));
  render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Click feedback for Movie ID 99" }));
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent(message);
  expect(alert).not.toHaveTextContent("sensitive upstream");
  expect(alert).toHaveFocus();
});

it("rejects a malformed successful response instead of fabricating confirmation", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ status: "almost" }, { status: 201 }));
  render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Feedback could not be recorded. Please try again.");
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();
});

it("requires the authoritative 201 status even with a recorded response body", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ status: "recorded" }, { status: 200 }));
  render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Feedback could not be recorded. Please try again.");
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();
});

it("redirects a 401 without a competing local error", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "Not authenticated" }, { status: 401 }));
  render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  await act(async () => undefined);
  expect(replace).toHaveBeenCalledWith("/login");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("aborts on event replacement and does not share state for the same movie across events", async () => {
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(pending.promise);
  const { rerender } = render(<RecommendationList result={result} />);
  await userEvent.click(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" }));
  const oldSignal = vi.mocked(fetch).mock.calls[0]?.[1]?.signal;

  rerender(<RecommendationList result={{ ...result, event_id: "4acffba7-1fad-4f2e-b542-095ac4f82039" }} />);
  expect(oldSignal?.aborted).toBe(true);
  expect(screen.getByRole("button", { name: "Record Like feedback for Movie ID 99" })).toBeEnabled();
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();

  pending.resolve(Response.json({ status: "recorded" }, { status: 201 }));
  await act(async () => undefined);
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();
});
