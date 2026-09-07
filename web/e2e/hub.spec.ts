/**
 * The launcher, in a real browser.
 *
 * There is little logic here to break; what these guard are the three things
 * that would make it lie or fail to do its job. The rating on the page must come
 * from the manifest rather than from a typed number, the button must actually
 * start a game, and the research the project rests on must be reachable without
 * being in a player's way.
 */

import { expect, test } from "@playwright/test";

test("the launcher leads with Reversi and its measured strength", async ({ page }) => {
  await page.goto("/");

  const reversi = page.locator('[data-game="reversi"]');
  await expect(reversi.getByRole("heading", { name: "Reversi" })).toBeVisible();
  await expect(reversi.getByText(/\d+ opponents/)).toBeVisible();
  await expect(reversi.getByText(/\+\d+/)).toBeVisible();
  await expect(page.getByRole("link", { name: "Play Reversi" })).toBeVisible();

  // The board on the page is a real position, so it holds more than the four
  // discs an opening has.
  expect(await reversi.locator("[data-square] span[data-disc]").count()).toBeGreaterThan(4);
});

test("Gomoku is present but plainly not playable", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Coming soon" })).toBeVisible();
  const gomoku = page.locator('[data-game="gomoku"]');
  await expect(gomoku.getByText("Gomoku")).toBeVisible();
  await expect(gomoku.getByRole("link")).toHaveCount(0);
});

test("the research is one click away and closed by default", async ({ page }) => {
  await page.goto("/");

  const details = page.locator("details", { hasText: "How the AI learned" });
  await expect(details.getByText(/round robin/)).toBeHidden();
  await details.getByText("How the AI learned").click();
  await expect(details.getByText(/round robin/)).toBeVisible();
});

test("Play Reversi opens the game, and the game is ready", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Play Reversi" }).click();

  await expect(page).toHaveURL(/\/reversi\/$/);
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 60_000 });
  // And the way back is on the page.
  await expect(page.getByRole("link", { name: /Reversi Zero, all games/ })).toBeVisible();
});
