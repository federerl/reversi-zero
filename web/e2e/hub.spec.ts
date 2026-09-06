/**
 * The front page, in a real browser.
 *
 * The hub is a few kilobytes of static content, so there is little to break;
 * what this guards is the two things that would make it lie. The card's rating
 * must come from the manifest rather than from a typed number, and Play must
 * actually land on a game that is ready to play.
 */

import { expect, test } from "@playwright/test";

test("the hub lists Reversi with a measured rating and a planned game marked as such", async ({
  page,
}) => {
  await page.goto("/");

  const reversi = page.locator('[data-game="reversi"]');
  await expect(reversi.getByRole("heading", { name: "Reversi" })).toBeVisible();
  await expect(reversi.getByText(/\+\d+ Elo/)).toBeVisible();
  await expect(reversi.getByRole("link", { name: "Play", exact: true })).toBeVisible();

  const gomoku = page.locator('[data-game="gomoku"]');
  await expect(gomoku.getByText(/not playable yet/i)).toBeVisible();
  await expect(gomoku.getByRole("link", { name: "Play" })).toHaveCount(0);
});

test("Play opens the game and the game is ready", async ({ page }) => {
  await page.goto("/");
  await page.locator('[data-game="reversi"]').getByRole("link", { name: "Play", exact: true }).click();

  await expect(page).toHaveURL(/\/reversi\/$/);
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 60_000 });
  // And the way back is on the page.
  await expect(page.getByRole("link", { name: "All games" })).toBeVisible();
});
