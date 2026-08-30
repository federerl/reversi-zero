/**
 * A game, in a real browser, end to end.
 *
 * The unit tests prove the engine agrees with Python. They cannot catch the
 * failures that live in the wiring, and the wiring is where the worst bug in
 * this build so far actually was: the effect that drives the agent's turn
 * listed the "thinking" flag as a dependency, and its first act is to set it --
 * so React tore the effect down and its cleanup aborted the search that had
 * just started. The agent cancelled itself on every move and sat at
 * "Thinking…" forever.
 *
 * Every assertion in this file would have failed on that build, and none of the
 * 35 unit tests did.
 */

import { expect, test, type Page } from "@playwright/test";

/** Squares the board is currently offering, in board index order. */
async function playableSquares(page: Page): Promise<string[]> {
  return page.locator("[data-square]:not([disabled])").evaluateAll((nodes) =>
    nodes.map((node) => (node as HTMLElement).dataset["square"]!),
  );
}

async function discCount(page: Page): Promise<number> {
  // `data-disc`, not "a round span": the legal-move dots are round spans too,
  // and counting those made the opening position look like it had eight discs.
  return page.locator("[data-square] span[data-disc]").count();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  // Loading a network is a download plus a compile; the status line says so
  // until it is ready, and clicking before then would be testing nothing.
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 60_000 });
});

test("the opening position is the standard one", async ({ page }) => {
  expect(await discCount(page)).toBe(4);
  // d3, c4, f5, e6 -- the four openings available to black, every game.
  expect((await playableSquares(page)).sort()).toEqual(["19", "26", "37", "44"]);
});

test("the agent answers a move, and the game moves on", async ({ page }) => {
  await page.locator('[data-square="19"]').click();

  // It thinks, and then it does not.
  await expect(page.getByRole("status")).toContainText("Thinking", { timeout: 5_000 });
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 30_000 });

  // Black played one disc and flipped one; white then played and flipped.
  expect(await discCount(page)).toBeGreaterThan(4);

  // And it reports what it did, rather than only that it did something.
  await expect(page.getByText(/played \w+\d · \d+ sims · \d+ ms/)).toBeVisible();
});

test("the board is never clickable while the agent is thinking", async ({ page }) => {
  // Otherwise a fast player can queue a move into a position that no longer
  // exists by the time it lands.
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Thinking");
  expect(await playableSquares(page)).toEqual([]);
});

test("a whole game can be played to the end", async ({ page }) => {
  // Casual, so this finishes in a reasonable time. What is being checked is
  // that the game terminates properly -- including any forced passes along the
  // way, which is where a rules port most often goes wrong.
  await page.getByLabel("Thinking time").selectOption("casual");

  for (let move = 0; move < 70; move++) {
    const status = await page.getByRole("status").textContent();
    if (status?.includes("win") || status?.includes("draw")) break;

    if (status?.includes("must pass")) {
      await page.getByRole("button", { name: "Pass" }).click();
    } else {
      const squares = await playableSquares(page);
      if (squares.length === 0) {
        await page.waitForTimeout(200);
        continue;
      }
      await page.locator(`[data-square="${squares[0]}"]`).click();
    }
    await expect(page.getByRole("status")).not.toContainText("Thinking", { timeout: 30_000 });
  }

  // A finished game says who won and by how much.
  await expect(page.getByRole("status")).toContainText(/win|draw/, { timeout: 30_000 });
  expect(await discCount(page)).toBeGreaterThan(50);
});

test("taking a move back returns the board to the player", async ({ page }) => {
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 30_000 });
  const afterOneMove = await discCount(page);

  await page.getByRole("button", { name: "Take back" }).click();

  expect(await discCount(page)).toBeLessThan(afterOneMove);
  expect((await playableSquares(page)).sort()).toEqual(["19", "26", "37", "44"]);
});

test("the opponent is labelled with a measured rating", async ({ page }) => {
  // The repository's rule: difficulty labels state measured strength, never
  // adjectives. This is that rule, asserted where a reader would see it.
  await expect(page.getByLabel("Opponent")).toContainText(/Generation \d+ — \d+ Elo/);
  await expect(page.getByText(/95% interval \d+–\d+, random play = 0/)).toBeVisible();
});

test("the board can be played with the keyboard alone", async ({ page }) => {
  await page.locator('[data-square="19"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 30_000 });
  expect(await discCount(page)).toBeGreaterThan(4);
});
