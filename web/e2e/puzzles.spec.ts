/**
 * The puzzle page in a real browser.
 *
 * The unit tests prove the curriculum is sound and the reducer does what it
 * says. What they cannot see is the wiring: whether the page mounts at all,
 * whether a click reaches the reducer, whether the verdict appears, and whether
 * a reload brings progress back. The puzzles page loads no network and runs no
 * search, so unlike the game page it either works immediately or is broken.
 *
 * It also has no engine to wait for, which makes a stall here meaningful rather
 * than slow: if the board is not interactive on arrival, something is wrong.
 */

import { expect, test, type Page } from "@playwright/test";

const consoleLog: string[] = [];

/** Squares the page is currently offering, in board index order. */
async function playableSquares(page: Page): Promise<string[]> {
  return page.locator("[data-square]:not([disabled])").evaluateAll((nodes) =>
    nodes.map((node) => (node as HTMLElement).dataset["square"]!),
  );
}

test.beforeEach(async ({ page }) => {
  consoleLog.length = 0;
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      consoleLog.push(`[${message.type()}] ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleLog.push(`[pageerror] ${error.message}`));

  await page.goto("/puzzles/");
  await expect(page.getByRole("status")).toContainText("to play, and winning");
});

test.afterEach(() => {
  // The data file is decoded at module load and throws loudly if it is not
  // usable. A page that rendered anyway while logging that is a page whose
  // curriculum silently came from nowhere.
  expect(consoleLog).toEqual([]);
});

test("it opens on stage one with a position to solve", async ({ page }) => {
  await expect(page.getByRole("button", { name: /^Stage 1/ })).toHaveAttribute(
    "aria-current",
    "true",
  );
  // Stage 1 is four or five empty squares, so the board offers very few moves
  // and every one of them is a real choice.
  const offered = await playableSquares(page);
  expect(offered.length).toBeGreaterThanOrEqual(2);
  expect(offered.length).toBeLessThanOrEqual(6);
});

test("an answer is graded immediately, with the exact margin", async ({ page }) => {
  const offered = await playableSquares(page);
  await page.locator(`[data-square="${offered[0]}"]`).click();

  // Either verdict is exact and says so in discs. No "wrong", and no waiting:
  // the answer was computed before the page was built.
  await expect(page.getByRole("status")).toContainText(/wins by \d+|loses by \d+|draws/);

  // And the board stops taking clicks once it has been answered.
  expect(await playableSquares(page)).toEqual([]);
});

test("a wrong answer can be retried, and the winning line can be watched", async ({ page }) => {
  const offered = await playableSquares(page);
  await page.locator(`[data-square="${offered[0]}"]`).click();

  await page.getByRole("button", { name: "Show the winning line" }).click();
  await expect(page.getByRole("status")).toContainText("Perfect play from here");

  // Stepping through returns the board to the puzzle position first, then walks
  // the stored line out to the end of the game.
  await page.getByRole("button", { name: "Next move" }).click();
  await expect(page.getByRole("status")).toContainText(/of \d+ moves/);

  await page.getByRole("button", { name: "Back to the puzzle" }).click();
  await expect(page.getByRole("status")).toContainText("to play, and winning");
  expect((await playableSquares(page)).length).toBeGreaterThan(0);
});

test("every move's exact result is one click away", async ({ page }) => {
  const offered = await playableSquares(page);
  await page.getByText("Every move, and where it leads").click();

  // One row per legal move, so the ranking is the whole table and not a
  // shortlist -- telling somebody their move was wrong is worth little next to
  // showing them where each move actually ends.
  const rows = page.getByRole("listitem");
  await expect(rows).toHaveCount(offered.length);
  for (const row of await rows.all()) {
    await expect(row).toContainText(/wins by \d+|loses by \d+|draws/);
  }

  // At least one row is marked best, and the table is ordered best first.
  await expect(rows.filter({ hasText: "best" }).first()).toBeVisible();
  await expect(rows.first()).toContainText("best");
});

test("a stage can be chosen, and every stage has puzzles", async ({ page }) => {
  const stages = page.getByRole("navigation", { name: "Stages" }).getByRole("button");
  const count = await stages.count();
  expect(count).toBeGreaterThanOrEqual(3);

  await stages.nth(count - 1).click();
  await expect(stages.nth(count - 1)).toHaveAttribute("aria-current", "true");
  await expect(page.getByRole("status")).toContainText("to play, and winning");
  expect((await playableSquares(page)).length).toBeGreaterThanOrEqual(2);
});

test("a solved puzzle is still solved after a reload", async ({ page }) => {
  const before = page.getByRole("button", { name: /^Stage 1/ });
  await expect(before).toContainText("0/");

  // Find the winning move by reading the page's own answer table rather than
  // guessing: the puzzle that comes first depends on the mined curriculum.
  await page.getByText("Every move, and where it leads").click();
  const bestSquare = await page
    .getByRole("listitem")
    .filter({ hasText: "best" })
    .first()
    .innerText();
  const name = bestSquare.trim().split(/\s+/)[0]!;
  const square = (Number(name.slice(1)) - 1) * 8 + "abcdefgh".indexOf(name[0]!);

  await page.locator(`[data-square="${square}"]`).click();
  await expect(page.getByRole("status")).toContainText(/wins by \d+/);
  await expect(before).toContainText("1/");

  await page.reload();
  await expect(page.getByRole("button", { name: /^Stage 1/ })).toContainText("1/");
});

test("the hub offers a way in", async ({ page }) => {
  await page.goto("/");
  const link = page.getByRole("link", { name: "Endgame puzzles" });
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/\/puzzles\/$/);
  await expect(page.getByRole("status")).toContainText("to play, and winning");
});
