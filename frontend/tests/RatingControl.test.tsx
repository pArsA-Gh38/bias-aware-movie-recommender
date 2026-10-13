import { render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { RatingControl } from "@/components/RatingControl";

it("renders exactly five visible numbered buttons with exact labels", () => {
  render(<RatingControl pending={false} onChange={vi.fn()} />);
  const group = screen.getByRole("group", { name: "Rate this movie" });
  const buttons = within(group).getAllByRole("button");
  expect(buttons).toHaveLength(5);
  buttons.forEach((button, index) => {
    expect(button).toHaveAccessibleName(`Rate ${index + 1} out of 5`);
    expect(button).toHaveTextContent(String(index + 1));
    expect(button).toHaveAttribute("aria-pressed", "false");
  });
});

it("uses the dynamic rated legend and a visible non-color selection indicator", () => {
  render(<RatingControl value={4} pending={false} onChange={vi.fn()} />);
  const group = screen.getByRole("group", { name: "Your rating: 4" });
  const selected = within(group).getByRole("button", { name: "Rate 4 out of 5" });
  expect(selected).toHaveAttribute("aria-pressed", "true");
  expect(selected).toHaveTextContent("4");
  expect(selected).toHaveTextContent("✓");
});

it("uses the saving legend and disables the fieldset while pending", () => {
  render(<RatingControl value={5} pending onChange={vi.fn()} />);
  expect(screen.getByRole("group", { name: "Saving rating…" })).toBeDisabled();
});
