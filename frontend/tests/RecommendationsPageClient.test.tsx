import { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { RecommendationsPageClient } from "@/components/RecommendationsPageClient";
import type { RecommendationResponse } from "@/lib/api/types";

const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace }),
}));

const result: RecommendationResponse = {
  event_id: "event-server",
  model_version_id: "model-server",
  strategy: "als_fold_in",
  configuration_key: "A0_L1",
  mode: "base",
  lambda: 1,
  items: [{ movie_id: 367, title: "Authoritative Movie 367", genres: ["Drama", "Synthetic"], rank: 1, score: 0.81234567 }],
};

function success(overrides: Partial<typeof result> = {}) {
  return Response.json({ ...result, ...overrides });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function requestBody(callIndex = 0) {
  const init = vi.mocked(fetch).mock.calls[callIndex]?.[1];
  return JSON.parse(String(init?.body)) as Record<string, unknown>;
}

beforeEach(() => {
  replace.mockReset();
  vi.stubGlobal("fetch", vi.fn());
});

it("makes no request on initial mount under StrictMode and selects A0_L1 by default", () => {
  render(<StrictMode><RecommendationsPageClient /></StrictMode>);

  expect(fetch).not.toHaveBeenCalled();
  expect(screen.getByRole("radio", { name: /^Relevance first \(default\)/ })).toBeChecked();
});

it.each([
  ["Relevance first \\(default\\)", "A0_L1"],
  ["Diversity — λ 0\\.7", "A1_L07"],
  ["Stronger diversity — λ 0\\.6", "A1_L06"],
] as const)("selects %s without requesting and submits only its key", async (labelPattern, key) => {
  vi.mocked(fetch).mockResolvedValue(success());
  render(<RecommendationsPageClient />);

  const radio = screen.getByRole("radio", { name: new RegExp(`^${labelPattern}`) });
  await userEvent.click(radio);
  expect(radio).toBeChecked();
  expect(fetch).not.toHaveBeenCalled();

  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByText("Movie ID 367");

  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch).toHaveBeenCalledWith(
    "/api/recommendations",
    expect.objectContaining({ method: "POST" }),
  );
  expect(requestBody()).toEqual({ configuration_key: key });
  expect(requestBody()).not.toHaveProperty("mode");
  expect(requestBody()).not.toHaveProperty("lambda");
  expect(requestBody()).not.toHaveProperty("lambda_");
  expect(requestBody()).not.toHaveProperty("beta");
});

it("keeps native radio keyboard behavior and exposes the legend, labels, and descriptions", async () => {
  render(<RecommendationsPageClient />);
  expect(screen.getByRole("group", { name: "Recommendation style" })).toBeInTheDocument();
  const relevance = screen.getByRole("radio", { name: /^Relevance first/ });
  const diversity = screen.getByRole("radio", { name: /^Diversity — λ 0.7/ });
  expect(relevance).toHaveAccessibleDescription("Prioritizes the model's relevance ranking without a diversity re-ranking adjustment.");
  expect(diversity).toHaveAccessibleDescription("Balances relevance with more variety across the ranked results.");

  relevance.focus();
  await userEvent.keyboard("{ArrowDown}");
  expect(diversity).toBeChecked();
});

it("uses a synchronous lock for rapid duplicate submissions and disables controls", async () => {
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValue(pending.promise);
  render(<RecommendationsPageClient />);
  const form = screen.getByRole("button", { name: "Generate recommendations" }).closest("form")!;

  fireEvent.submit(form);
  fireEvent.submit(form);

  expect(fetch).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("button", { name: "Generating…" })).toBeDisabled();
  for (const radio of screen.getAllByRole("radio")) expect(radio).toBeDisabled();
  expect(screen.getByRole("status")).toHaveTextContent("Generating recommendations…");

  pending.resolve(success());
  expect(await screen.findByText("Movie ID 367")).toBeInTheDocument();
});

it("clears a prior result when a new request begins", async () => {
  vi.mocked(fetch).mockResolvedValueOnce(success());
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByText("Movie ID 367");

  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(pending.promise);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  expect(screen.queryByText("Movie ID 367")).not.toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Generating recommendations…");
  pending.resolve(success());
  await screen.findByText("Movie ID 367");
});

it("retains successful results as stale after selection and clears staleness after regeneration", async () => {
  vi.mocked(fetch)
    .mockResolvedValueOnce(success())
    .mockResolvedValueOnce(success({
      configuration_key: "A1_L07",
      mode: "diversity_only",
      lambda: 0.7,
    }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByText("Movie ID 367");

  await userEvent.click(screen.getByRole("radio", { name: /^Diversity — λ 0.7/ }));
  expect(screen.getByText("Movie ID 367")).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent(
    "These results use Relevance first (default). Generate again to apply Diversity — λ 0.7.",
  );
  expect(fetch).toHaveBeenCalledTimes(1);

  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await waitFor(() => expect(screen.queryByText(/These results use/)).not.toBeInTheDocument());
  expect(requestBody(1)).toEqual({ configuration_key: "A1_L07" });
  expect(screen.getByText("diversity_only")).toBeInTheDocument();
});

it("renders server-confirmed applied metadata instead of reconstructing selected values", async () => {
  vi.mocked(fetch).mockResolvedValue(success({
    configuration_key: "A1_L06",
    mode: "diversity_only",
    lambda: 0.6,
  }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  expect(await screen.findByText("A1_L06")).toBeInTheDocument();
  expect(screen.getByText("diversity_only")).toBeInTheDocument();
  expect(screen.getByText("0.6")).toBeInTheDocument();
  expect(screen.getByText("event-server")).toBeInTheDocument();
  expect(screen.getByText("model-server")).toBeInTheDocument();
  expect(screen.getByText("als_fold_in")).toBeInTheDocument();
});

it("keeps applied metadata for an empty successful response and focuses the status heading", async () => {
  vi.mocked(fetch).mockResolvedValue(success({ items: [] }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  const heading = await screen.findByRole("heading", { name: "Recommendation status" });
  expect(screen.getByText("No recommendations are available yet.")).toBeInTheDocument();
  expect(screen.getByText("event-server")).toBeInTheDocument();
  expect(heading).toHaveFocus();
});

it("focuses the results heading after success", async () => {
  vi.mocked(fetch).mockResolvedValue(success());
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  expect(await screen.findByRole("heading", { name: "Your ranked recommendations" })).toHaveFocus();
});

it("shows exact cold-start guidance and the movies link", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "upstream detail" }, { status: 409 }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  const heading = await screen.findByRole("heading", { name: "Rate a few more movies" });
  expect(screen.getByText("At least five ratings of 4 or 5 are required.")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Continue rating movies" })).toHaveAttribute("href", "/movies");
  expect(heading).toHaveFocus();
});

it.each([
  [422, "The selected recommendation style was rejected. Choose one of the available styles and try again."],
  [503, "The recommendation service is temporarily unavailable. Please try again."],
  [504, "The recommendation service is temporarily unavailable. Please try again."],
  [500, "Could not generate recommendations. Please try again."],
] as const)("shows the sanitized retryable error for status %s and focuses its heading", async (status, message) => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "sensitive upstream detail" }, { status }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent(message);
  expect(alert).not.toHaveTextContent("sensitive upstream detail");
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Recommendations could not be generated" })).toHaveFocus();
});

it("retries through the shared generation function using the current selection", async () => {
  vi.mocked(fetch)
    .mockResolvedValueOnce(Response.json({ detail: "rejected" }, { status: 422 }))
    .mockResolvedValueOnce(success({
      configuration_key: "A1_L06",
      mode: "diversity_only",
      lambda: 0.6,
    }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByRole("alert");
  await userEvent.click(screen.getByRole("radio", { name: /^Stronger diversity — λ 0.6/ }));
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));

  expect(await screen.findByText("A1_L06")).toBeInTheDocument();
  expect(requestBody(1)).toEqual({ configuration_key: "A1_L06" });
});

it("redirects a 401 only to login", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "Not authenticated" }, { status: 401 }));
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));

  await act(async () => undefined);
  expect(replace).toHaveBeenCalledTimes(1);
  expect(replace).toHaveBeenCalledWith("/login");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("keeps stale-result feedback usable and isolates a regenerated event from a late response", async () => {
  const feedback = deferred<Response>();
  vi.mocked(fetch).mockImplementation((input) => {
    if (input === "/api/feedback") return feedback.promise;
    return Promise.resolve(success());
  });
  render(<RecommendationsPageClient />);
  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByText("Movie ID 367");

  await userEvent.click(screen.getByRole("radio", { name: /^Diversity — λ 0.7/ }));
  const like = screen.getByRole("button", { name: "Record Like feedback for Movie ID 367" });
  expect(like).toBeEnabled();
  fireEvent.click(like);
  const feedbackCall = vi.mocked(fetch).mock.calls.find(([url]) => url === "/api/feedback")!;

  await userEvent.click(screen.getByRole("button", { name: "Generate recommendations" }));
  await screen.findByText("Movie ID 367");
  expect(feedbackCall[1]?.signal?.aborted).toBe(true);
  expect(screen.getByRole("button", { name: "Record Like feedback for Movie ID 367" })).toBeEnabled();

  feedback.resolve(Response.json({ status: "recorded" }, { status: 201 }));
  await act(async () => undefined);
  expect(screen.queryByText(/Recorded Like/)).not.toBeInTheDocument();
});
