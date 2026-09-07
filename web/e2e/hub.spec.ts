/**
 * The launcher, in a real browser.
 *
 * There is little logic here to break; what these guard are the three things
 * that would make it lie or fail to do its job. The level count on the page has
 * to be the number of levels a visitor can really play, the button has to
 * actually start a game, and the research the project rests on has to be
 * reachable without being in a player's way.
 */

import { expect, test } from "@playwright/test";

test("the launcher leads with Reversi and the levels it offers", async ({ page }) => {
  await page.goto("/");

  const reversi = page.locator('[data-game="reversi"]');
  await expect(reversi.getByRole("heading", { name: "Reversi" })).toBeVisible();
  await expect(reversi.getByText(/^\w+ levels, \w+ to \w+$/)).toBeVisible();
  await expect(page.getByRole("link", { name: "Play Reversi" })).toBeVisible();

  // A rating is a fact about the project, not the first thing to tell somebody
  // deciding whether to play. It belongs in the disclosure, and this pins that
  // it is not on the page beside the button.
  await expect(reversi.getByText(/\+\d+/)).toBeHidden();

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
  // And the strongest rating is in there, from the manifest, not typed.
  await expect(details.getByText(/top level rates about \+\d+/)).toBeVisible();
});

test("Play Reversi opens the game, and the game is ready", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Play Reversi" }).click();

  await expect(page).toHaveURL(/\/reversi\/$/);
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 60_000 });
  // And the way back is on the page.
  await expect(page.getByRole("link", { name: /Reversi Zero, all games/ })).toBeVisible();
});
