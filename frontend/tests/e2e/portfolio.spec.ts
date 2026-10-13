import { expect, test, type Page } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

async function capture(page: Page, name: string) {
  if (!process.env.CAPTURE_PORTFOLIO_SCREENSHOTS) return;
  await mkdir(path.resolve("../docs/screenshots"), { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `../docs/screenshots/${name}.png` });
}

test("register, rate, recommend, record feedback and inspect research evidence", async ({ page }) => {
  await page.goto("/register");
  await page.getByLabel("Email", { exact: true }).fill(`demo-${Date.now() % 100000}@example.com`);
  await page.getByLabel("Password", { exact: true }).fill("Synthetic-demo-only-2026");
  await page.getByRole("button", { name: "Register", exact: true }).click();
  await expect(page).toHaveURL(/\/movies$/);
  await expect(page.getByText("Synthetic demo · fictional movies")).toBeVisible();
  for (let index = 0; index < 5; index++) {
    const card = page.locator("article").nth(index);
    const saved = page.waitForResponse(response =>
      response.request().method() === "PUT" && response.url().includes("/api/ratings/"));
    await card.getByRole("button", { name: "Rate 5 out of 5", exact: true }).click();
    expect((await saved).ok()).toBe(true);
    await expect(card.getByRole("button", { name: "Rate 5 out of 5", exact: true })).toHaveAttribute("aria-pressed", "true");
  }
  await expect(page.getByText("Saving rating…", { exact: true })).toHaveCount(0);
  await capture(page, "catalog");
  await page.getByRole("link", { name: "Recommendations", exact: true }).click();
  await page.getByRole("button", { name: "Generate recommendations", exact: true }).click();
  await expect(page.getByRole("list", { name: "Ranked movie recommendations" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Ranked movie recommendations" }).locator("li")).toHaveCount(10);
  await page.getByRole("button", { name: /Record Like feedback/ }).first().click();
  await expect(page.getByText(/Recorded Like for Movie ID/)).toBeVisible();
  await capture(page, "recommendations");
  await page.getByRole("link", { name: "Experiments", exact: true }).click();
  await expect(page.getByText(/NDCG@10/).first()).toBeVisible();
  await capture(page, "experiments");
});
