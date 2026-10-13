import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { MovieCard } from "@/components/MovieCard";

it("renders canonical catalog metadata, all genres, and an aria-hidden decoration", () => {
  const { container } = render(<MovieCard movie={{ movie_id: 42, title: "A Very Long Example", release_year: 2001, genres: ["Drama", "Comedy"] }} onRate={vi.fn()} />);
  expect(screen.getByRole("heading", { name: "A Very Long Example" })).toBeInTheDocument();
  expect(screen.getByText("Movie #42 · 2001")).toBeInTheDocument();
  expect(within(screen.getByRole("list", { name: "Genres" })).getAllByRole("listitem")).toHaveLength(2);
  expect(container.querySelector("[aria-hidden='true']")).toBeInTheDocument();
});

it("omits a nullable release year without inventing metadata", () => {
  render(<MovieCard movie={{ movie_id: 7, title: "Example", release_year: null, genres: [] }} onRate={vi.fn()} />);
  expect(screen.getByText("Movie #7")).toBeInTheDocument();
  expect(screen.queryByRole("list", { name: "Genres" })).not.toBeInTheDocument();
});

it("isolates pending state to the card whose asynchronous rating is unresolved", async () => {
  let resolve!: () => void;
  const pending = new Promise<void>((done) => { resolve = done; });
  render(<>
    <MovieCard movie={{ movie_id: 1, title: "First", release_year: null, genres: [] }} onRate={() => pending} />
    <MovieCard movie={{ movie_id: 2, title: "Second", release_year: null, genres: [] }} onRate={vi.fn().mockResolvedValue(undefined)} />
  </>);
  fireEvent.click(screen.getAllByRole("button", { name: "Rate 4 out of 5" })[0]);
  expect(screen.getAllByRole("group")[0]).toBeDisabled();
  expect(screen.getAllByRole("button", { name: "Rate 4 out of 5" })[1]).toBeEnabled();
  await act(async () => resolve());
  expect(await screen.findAllByText("Rate this movie")).toHaveLength(2);
});

it("retains the backend error and retry guidance, then permits another rating", async () => {
  const onRate = vi.fn().mockRejectedValueOnce(new Error("Rating service unavailable")).mockResolvedValueOnce(undefined);
  render(<MovieCard movie={{ movie_id: 42, title: "Example", release_year: null, genres: [] }} onRate={onRate} />);
  await userEvent.click(screen.getByRole("button", { name: "Rate 5 out of 5" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Rating service unavailable Choose a rating to retry.");
  await userEvent.click(screen.getByRole("button", { name: "Rate 4 out of 5" }));
  expect(onRate).toHaveBeenLastCalledWith(42, 4);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});
