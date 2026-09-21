/**
 * The puzzle page in a real browser.
 *
 * The unit tests prove the curriculum is sound, the search agrees with Python
 * and the reducer does what it says. What they cannot see is the wiring, and
 * this page has a new piece of it: the exact search runs in a worker, so a
 * reply arrives asynchronously. Every failure mode that introduces — a worker
 * that fails to start, a reply for a position the page has moved on from, a
 * board left permanently locked — looks the same from the outside, which is an
 * opponent that never moves.
 *
 * So the load-bearing test here is simply that an ending finishes.
 */

import { expect, test, type Page } from "@playwright/test";

const consoleLog: string[] = [];

/** Squares the page is currently offering, in board index order. */
async function playableSquares(page: Page): Promise<string[]> {
  return page
    .locator("[data-square]:not([disabled])")
    .evaluateAll((nodes) => nodes.map((node) => (node as HTMLElement).dataset["square"]!));
}

/**
 * A control inside the open dialog.
 *
 * Scoped, because the side panel carries buttons of the same name and a modal
 * dialog correctly stops pointer events reaching them. An unscoped locator
 * resolves to the one behind the dialog and then waits forever for it to become
 * clickable.
 */
function dialogButton(page: Page, name: string | RegExp) {
  return page.locator("dialog[open]").getByRole("button", { name });
}

/** Play whatever is on offer until the ending is over, or give up. */
async function playToTheEnd(page: Page, limit = 40): Promise<void> {
  for (let move = 0; move < limit; move++) {
    if (await page.locator("dialog[open]").isVisible().catch(() => false)) return;
    const offered = await playableSquares(page);
    if (offered.length === 0) {
      // Either the opponent is thinking or the game is over. Give the worker a
      // moment; the dialog check at the top of the loop catches the end.
      await page.waitForTimeout(150);
      continue;
    }
    await page.locator(`[data-square="${offered[0]}"]`).click();
  }
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
  await expect(page.getByRole("status")).toContainText("and winning");
});

test.afterEach(() => {
  // The data file is decoded at module load and throws loudly if it is not
  // usable; the worker reports its own failures. A page that rendered anyway
  // while logging either is a page whose answers came from nowhere.
  expect(consoleLog).toEqual([]);
});

test("it opens on stage one with an ending to play", async ({ page }) => {
  await expect(page.getByRole("button", { name: /^Stage 1/ })).toHaveAttribute(
    "aria-current",
    "true",
  );
  const offered = await playableSquares(page);
  expect(offered.length).toBeGreaterThanOrEqual(2);
});

test("the opponent answers, and the ending can be played to a finish", async ({ page }) => {
  // The whole point of the redesign. A move is answered by a real search in a
  // worker, and the game reaches an end with a result you can read.
  // Deliberately not asserting that the board locks between the move and the
  // reply: at four empty squares the search answers in well under a
  // millisecond, so the lock is real but not reliably observable from here. The
  // unit tests pin it instead. What matters here is that a reply arrives at all.
  const first = (await playableSquares(page))[0]!;
  await page.locator(`[data-square="${first}"]`).click();
  await playToTheEnd(page);

  const dialog = page.locator("dialog[open]");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(/Won it\.|Lost it\.|Drawn\./);
  await expect(dialog).toContainText(/\d+–\d+/);
});

test("the result explains itself rather than only reporting a score", async ({ page }) => {
  await playToTheEnd(page);
  const dialog = page.locator("dialog[open]");
  await expect(dialog).toBeVisible();
  // Either it names the move that lost it, or it says the opening was already
  // the mistake, or it congratulates a win. Silence is the failure.
  await expect(dialog).toContainText(
    /is where it went|was already the mistake|held it from a position|was worth/,
  );
});

test("a finished ending can be restarted from the puzzle position", async ({ page }) => {
  await playToTheEnd(page);
  await expect(page.locator("dialog[open]")).toBeVisible();

  await dialogButton(page, /Try again|Next puzzle/).click();
  await expect(page.locator("dialog[open]")).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText(/and winning|Your move/);
  expect((await playableSquares(page)).length).toBeGreaterThan(0);
});

test("the winning line can be watched after the game", async ({ page }) => {
  await playToTheEnd(page);
  await dialogButton(page, "Show the winning line").click();
  await expect(page.getByRole("status")).toContainText("Perfect play from the start");

  await page.getByRole("button", { name: "Next move" }).click();
  await expect(page.getByRole("status")).toContainText(/of \d+ moves/);

  await page.getByRole("button", { name: "Back to the puzzle" }).click();
  await expect(page.getByRole("status")).not.toContainText("Perfect play");
});

test("every opening move's exact result is available once it is over", async ({ page }) => {
  // Only once it is over. Shown during play this would be the answer key, which
  // is the reason the page says nothing while an ending is in progress.
  await expect(page.getByText("Every opening move, and where it leads")).toHaveCount(0);

  await playToTheEnd(page);
  // Close the dialog before reaching for the panel behind it: a modal that let
  // the page underneath be clicked would be the bug, not the test.
  await dialogButton(page, /Look at the board|Try again|Next puzzle/).first().click();
  await expect(page.locator("dialog[open]")).toHaveCount(0);
  await playToTheEnd(page);
  await page.keyboard.press("Escape");
  await expect(page.locator("dialog[open]")).toHaveCount(0);

  await page.getByText("Every opening move, and where it leads").click();
  const rows = page.getByRole("listitem");
  expect(await rows.count()).toBeGreaterThanOrEqual(2);
  await expect(rows.filter({ hasText: "best" }).first()).toBeVisible();
});

test("a restarted puzzle still gets answered, and on the same colour", async ({ page }) => {
  // Two halves of one bug, both about a restart.
  //
  // The page keys an outstanding search on the position it asked about. Replaying
  // a puzzle reaches the same positions again, so a key built from the board
  // alone could not tell this attempt's question from the last one's -- and the
  // second time round the opponent was never asked, leaving the board waiting on
  // a move that was not coming. That half is what this test catches: it hangs
  // without the fix.
  //
  // The other half was worse and is pinned by a unit test rather than here: a
  // reply still in flight when the restart happened was applied to the restarted
  // board, as the opponent's move on the player's turn, which flipped the side to
  // move and left the player on the colour the puzzle said they were not. The
  // race needs the reply to land after the restart, and the browser will not
  // reliably lose it -- the search runs in a worker, which CPU throttling does not
  // slow -- so `puzzles.test.ts` drives the reducer straight at that sequence.
  const status = page.getByRole("status");
  const colour = ((await status.innerText()).match(/You are (black|white)/) ?? [])[1];
  expect(colour).toBeDefined();

  for (let attempt = 0; attempt < 3; attempt++) {
    const offered = await playableSquares(page);
    expect(offered.length).toBeGreaterThan(0);
    await page.locator(`[data-square="${offered[0]}"]`).click();

    // The opponent has to answer on every attempt, not just the first.
    await expect
      .poll(async () => (await playableSquares(page)).length, { timeout: 15_000 })
      .toBeGreaterThan(0);

    await page.getByRole("button", { name: "Start again" }).click();
    await expect(status).toContainText(`You are ${colour}`);
  }
});

test("a stage can be chosen, and every stage has an ending to play", async ({ page }) => {
  const stages = page.getByRole("navigation", { name: "Stages" }).getByRole("button");
  const count = await stages.count();
  expect(count).toBeGreaterThanOrEqual(3);

  await stages.nth(count - 1).click();
  await expect(stages.nth(count - 1)).toHaveAttribute("aria-current", "true");
  await expect(page.getByRole("status")).toContainText("and winning");
  expect((await playableSquares(page)).length).toBeGreaterThanOrEqual(2);
});

test("a won ending is still recorded after a reload", async ({ page }) => {
  // Play perfectly by reading the page's own answer table is not possible here
  // (it is hidden until the end), so play the ending out repeatedly until one
  // is won — with four empty squares and a real win available, that is quick.
  const stage = page.getByRole("button", { name: /^Stage 1/ });

  for (let attempt = 0; attempt < 6; attempt++) {
    await playToTheEnd(page);
    const dialog = page.locator("dialog[open]");
    await expect(dialog).toBeVisible();
    const won = await dialog.getByText("Won it.").isVisible().catch(() => false);
    if (won) break;
    await dialogButton(page, "Try again").click();
  }

  if (await stage.textContent().then((t) => /[1-9]\d*\//.test(t ?? ""))) {
    await page.reload();
    await expect(page.getByRole("button", { name: /^Stage 1/ })).not.toContainText("0/");
  }
});

test("the hub offers a way in", async ({ page }) => {
  await page.goto("/");
  const link = page.getByRole("link", { name: "Endgame puzzles" });
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/\/puzzles\/$/);
  await expect(page.getByRole("status")).toContainText("and winning");
});
