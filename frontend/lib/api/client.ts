import type {
  Credentials,
  FeedbackRequest,
  FeedbackResponse,
  MovieListResponse,
  RatingResponse,
  RecommendationRequest,
  RecommendationResponse,
  ExperimentSummaryResponse,
  User,
  UserRatingsResponse,
} from "./types";
import { parseExperimentSummary } from "./experiment-summary";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

const BFF_REQUEST_TIMEOUT_MS = 15_000;

async function request<T>(path: string, init?: RequestInit, expectedStatus?: number): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort("timeout"), BFF_REQUEST_TIMEOUT_MS);
  const abort = () => controller.abort(init?.signal?.reason);
  if (init?.signal?.aborted) abort();
  else init?.signal?.addEventListener("abort", abort, { once: true });
  try {
    const response = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      signal: controller.signal,
    });

    if (!response.ok || (expectedStatus !== undefined && response.status !== expectedStatus)) {
      let body: { detail?: string | Array<{ msg?: string }> } = {};
      try {
        body = await response.json() as typeof body;
      } catch (caught) {
        if (controller.signal.aborted) throw caught;
      }
      const detail = Array.isArray(body.detail)
        ? body.detail.map((item) => item.msg).filter(Boolean).join(". ")
        : body.detail;
      throw new ApiError(detail || "Request failed. Please try again.", response.status);
    }

    return await response.json() as T;
  } catch (caught) {
    if (caught instanceof ApiError) throw caught;
    if (init?.signal?.aborted) throw caught;
    throw new ApiError("Application service unavailable. Please try again.", 503);
  } finally {
    clearTimeout(timeout);
    init?.signal?.removeEventListener("abort", abort);
  }
}

export const api = {
  login: (credentials: Credentials) =>
    request<User>("/api/auth/login", { method: "POST", body: JSON.stringify(credentials) }),
  register: (credentials: Credentials) =>
    request<User>("/api/auth/register", { method: "POST", body: JSON.stringify(credentials) }),
  me: () => request<User>("/api/auth/me"),
  logout: () => request<{ ok: true }>("/api/auth/logout", { method: "POST" }),
  movies: (query: URLSearchParams) => request<MovieListResponse>(`/api/movies?${query.toString()}`),
  ratings: () => request<UserRatingsResponse>("/api/ratings/me"),
  rate: (movieId: number, rating: number) =>
    request<RatingResponse>(`/api/ratings/${movieId}`, {
      method: "PUT",
      body: JSON.stringify({ rating }),
    }),
  recommendations: (recommendation: RecommendationRequest) =>
    request<RecommendationResponse>("/api/recommendations", {
      method: "POST",
      body: JSON.stringify({ configuration_key: recommendation.configuration_key }),
    }),
  feedback: async (feedback: FeedbackRequest, signal?: AbortSignal): Promise<FeedbackResponse> => {
    const response = await request<unknown>("/api/feedback", {
      method: "POST",
      body: JSON.stringify(feedback),
      signal,
    }, 201);
    if (
      typeof response !== "object"
      || response === null
      || Array.isArray(response)
      || Object.keys(response).length !== 1
      || (response as Record<string, unknown>).status !== "recorded"
    ) {
      throw new ApiError("Unexpected feedback response.", 502);
    }
    return response as FeedbackResponse;
  },
  experimentSummary: async (signal?: AbortSignal): Promise<ExperimentSummaryResponse> =>
    parseExperimentSummary(await request<unknown>("/api/experiments/summary", { signal })),
};
