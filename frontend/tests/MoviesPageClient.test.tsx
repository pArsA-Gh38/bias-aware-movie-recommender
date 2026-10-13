import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MoviesPageClient } from "@/components/MoviesPageClient";
import { ApiError, api } from "@/lib/api/client";
import type { MovieListResponse, RatingResponse } from "@/lib/api/types";

const replace = vi.fn();
const router = { replace };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("@/lib/api/client", async (loadOriginal) => {
  const original = await loadOriginal<typeof import("@/lib/api/client")>();
  return { ...original, api: { movies: vi.fn(), ratings: vi.fn(), rate: vi.fn() } };
});

const movie = (movie_id: number, title = `Movie ${movie_id}`) => ({ movie_id, title, release_year: 2000, genres: ["Drama"] });
const moviesMock = vi.mocked(api.movies);
const ratingsMock = vi.mocked(api.ratings);
const rateMock = vi.mocked(api.rate);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

beforeEach(() => {
  moviesMock.mockReset().mockResolvedValue({ items: [movie(1)], next_cursor: null });
  ratingsMock.mockReset().mockResolvedValue({ items: [] });
  rateMock.mockReset().mockResolvedValue({ movie_id: 1, rating: 4, positive_count: 1, updated_at: "now" });
  replace.mockReset();
});

it("shows the existing initial status while movies load and renders no main", async () => {
  const request = deferred<MovieListResponse>();
  moviesMock.mockReturnValueOnce(request.promise);
  const { container } = render(<MoviesPageClient />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading movies…");
  expect(container.querySelector("main")).not.toBeInTheDocument();
  request.resolve({ items: [movie(1)], next_cursor: null });
  expect(await screen.findByText("Movie 1")).toBeInTheDocument();
});

describe("onboarding progress", () => {
  it("shows plural copy and exact onboarding request parameters", async () => {
    render(<MoviesPageClient />);
    expect(await screen.findByText("5 more positive ratings needed")).toBeInTheDocument();
    expect(moviesMock.mock.calls[0][0].toString()).toBe("limit=24&onboarding=true");
  });

  it("shows singular remaining copy", async () => {
    ratingsMock.mockResolvedValueOnce({ items: [1, 2, 3, 4].map((movie_id) => ({ movie_id, rating: 4, created_at: "", updated_at: "" })) });
    render(<MoviesPageClient />);
    expect(await screen.findByText("1 more positive rating needed")).toBeInTheDocument();
  });

  it("unlocks recommendations at five positives", async () => {
    ratingsMock.mockResolvedValueOnce({ items: [1, 2, 3, 4, 5].map((movie_id) => ({ movie_id, rating: 5, created_at: "", updated_at: "" })) });
    render(<MoviesPageClient />);
    expect(await screen.findByText("You are ready for recommendations.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View recommendations" })).toHaveAttribute("href", "/recommendations");
  });
});

it("toggles catalog filters and submits trimmed title and genre with catalog parameters", async () => {
  render(<MoviesPageClient />);
  await screen.findByText("Movie 1");
  await userEvent.click(screen.getByRole("button", { name: "Browse full catalog" }));
  await waitFor(() => expect(moviesMock).toHaveBeenCalledTimes(2));
  expect(moviesMock.mock.calls[1][0].toString()).toBe("limit=24");
  await userEvent.type(screen.getByLabelText("Title"), "  Alien  ");
  await userEvent.type(screen.getByLabelText("Genre"), "  Sci-Fi  ");
  await userEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() => expect(moviesMock).toHaveBeenCalledTimes(3));
  expect(moviesMock.mock.calls[2][0].toString()).toBe("limit=24&q=Alien&genre=Sci-Fi");
  expect(screen.getByRole("button", { name: "Show onboarding picks" })).toBeInTheDocument();
});

it("sends the catalog cursor, retains existing cards while pending, and appends results", async () => {
  moviesMock.mockResolvedValueOnce({ items: [movie(1)], next_cursor: "next-page" });
  render(<MoviesPageClient />);
  await screen.findByText("Movie 1");
  moviesMock.mockResolvedValueOnce({ items: [movie(1)], next_cursor: "next-page" });
  await userEvent.click(screen.getByRole("button", { name: "Browse full catalog" }));
  await screen.findByRole("button", { name: "Load more" });
  const request = deferred<MovieListResponse>();
  moviesMock.mockReturnValueOnce(request.promise);
  await userEvent.click(screen.getByRole("button", { name: "Load more" }));
  expect(screen.getByText("Movie 1")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Loading…" })).toBeDisabled();
  expect(moviesMock.mock.calls.at(-1)?.[0].toString()).toBe("limit=24&cursor=next-page");
  request.resolve({ items: [movie(2)], next_cursor: null });
  expect(await screen.findByText("Movie 2")).toBeInTheDocument();
  expect(screen.getByText("Movie 1")).toBeInTheDocument();
});

it("redirects a movie-load 401 to login and keeps the backend error retryable", async () => {
  moviesMock.mockRejectedValueOnce(new ApiError("Session expired", 401)).mockResolvedValueOnce({ items: [movie(2)], next_cursor: null });
  render(<MoviesPageClient />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Session expired");
  expect(replace).toHaveBeenCalledWith("/login");
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText("Movie 2")).toBeInTheDocument();
});

it("uses backend positive_count after an optimistic rating update", async () => {
  rateMock.mockResolvedValueOnce({ movie_id: 1, rating: 4, positive_count: 4, updated_at: "now" });
  render(<MoviesPageClient />);
  await userEvent.click(await screen.findByRole("button", { name: "Rate 4 out of 5" }));
  expect(await screen.findByText("1 more positive rating needed")).toBeInTheDocument();
  expect(rateMock).toHaveBeenCalledWith(1, 4);
});

it("rolls a rejected optimistic rating back to the exact previous value and exposes retry", async () => {
  ratingsMock.mockResolvedValueOnce({ items: [{ movie_id: 1, rating: 2, created_at: "", updated_at: "" }] });
  const request = deferred<RatingResponse>();
  rateMock.mockReturnValueOnce(request.promise);
  render(<MoviesPageClient />);
  const four = await screen.findByRole("button", { name: "Rate 4 out of 5" });
  fireEvent.click(four);
  await waitFor(() => expect(four).toHaveAttribute("aria-pressed", "true"));
  await act(async () => request.reject(new Error("Backend rejected rating")));
  expect(await screen.findByRole("alert")).toHaveTextContent("Backend rejected rating Choose a rating to retry.");
  expect(screen.getByRole("button", { name: "Rate 2 out of 5" })).toHaveAttribute("aria-pressed", "true");
  expect(four).toHaveAttribute("aria-pressed", "false");
});

it("does not access browser token storage", async () => {
  const local = vi.spyOn(Storage.prototype, "setItem");
  const session = vi.spyOn(Storage.prototype, "getItem");
  render(<MoviesPageClient />);
  await screen.findByText("Movie 1");
  expect(local).not.toHaveBeenCalled();
  expect(session).not.toHaveBeenCalled();
  local.mockRestore();
  session.mockRestore();
});
