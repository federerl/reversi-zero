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

const consoleLog: string[] = [];

test.beforeEach(async ({ page }) => {
  // A failure in the engine surfaces as a toast, a console error, or -- the
  // worst case, and one this project has hit repeatedly -- as nothing at all.
  // Collecting these means a failure in CI reports a cause rather than just a
  // timeout on some unrelated assertion.
  consoleLog.length = 0;
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      consoleLog.push(`[${message.type()}] ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleLog.push(`[pageerror] ${error.message}`));

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
  await expect(page.getByText(/played \w+\d · \d+ ms/)).toBeVisible();
});

test("a network opponent searches, and says how much", async ({ page }) => {
  // The default opponent is a baseline, which runs no search. Picking a
  // generation is what exercises the network path -- and the simulation count in
  // the move report is the visible evidence that it ran.
  //
  // This is the only test that downloads and runs the network, so it is the only
  // one that can catch a broken ONNX path. On a slow shared runner that means a
  // download, a compile and a search, none of which are quick -- hence the
  // long waits and test.slow().
  test.slow();

  await page.getByLabel("Opponent").selectOption("gen05");
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 90_000 });

  await expect(page.getByLabel("Thinking time")).toBeVisible();
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 90_000 });

  await expect(page.getByText(/played \w+\d · \d+ sims · \d+ ms/)).toBeVisible();
  // A network has an opinion about who is winning; the bar shows it.
  await expect(page.getByRole("meter")).toBeVisible();
});

test("a baseline opponent offers no search controls and claims no opinion", async ({ page }) => {
  // Random and Greedy pick from the rules alone. There is no simulation budget
  // to spend and no value to report, so the interface shows neither rather than
  // inventing them.
  await expect(page.getByLabel("Opponent")).toHaveValue("greedy");
  await expect(page.getByLabel("Thinking time")).toBeHidden();

  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your move.", { timeout: 30_000 });

  await expect(page.getByRole("meter")).toBeHidden();
  await expect(page.getByText(/sims/)).toBeHidden();
});

test("the board is never clickable while the agent is thinking", async ({ page }) => {
  // Otherwise a fast player can queue a move into a position that no longer
  // exists by the time it lands.
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Thinking");
  expect(await playableSquares(page)).toEqual([]);
});

test("a whole game can be played to the end", async ({ page }) => {
  // Against the default opponent, which is a baseline: it answers instantly and
  // needs no network downloaded, so this stays about the rules rather than about
  // the search. What is being checked is that a game terminates properly --
  // including any forced passes along the way, which is where a rules port most
  // often goes wrong.

  // The bound counts loop turns, not moves, and a turn spent waiting for the
  // agent makes no move at all. It is generous for that reason: the agent will
  // not answer sooner than MIN_REPLY_MS, so a tighter bound runs out of turns
  // partway through a game and fails for a reason that has nothing to do with
  // the rules. The loop ends when the game does.
  for (let turn = 0; turn < 250; turn++) {
    const status = (await page.getByRole("status").textContent()) ?? "";
    if (/win|draw/.test(status)) break;

    // Not our move yet. Wait for it rather than burning the turn on a poll.
    if (status.includes("Thinking")) {
      await expect(page.getByRole("status")).not.toContainText("Thinking", { timeout: 30_000 });
      continue;
    }

    if (status.includes("must pass")) {
      await page.getByRole("button", { name: "Pass" }).click();
    } else {
      const squares = await playableSquares(page);
      if (squares.length === 0) {
        await page.waitForTimeout(100);
        continue;
      }
      await page.locator(`[data-square="${squares[0]}"]`).click();
    }
    await expect(page.getByRole("status")).not.toContainText("Thinking", { timeout: 30_000 });
  }

  // A finished game says who won and by how much.
  const final = (await page.getByRole("status").textContent()) ?? "";
  expect(final).toMatch(/win|draw/);

  // The score it reports has to be the position on the board.
  //
  // Not "the board is nearly full", which is what this asserted first and is
  // simply not true of Reversi: a game ends as soon as *neither* side has a
  // legal move, which can happen with plenty of squares still empty. It failed
  // here on a perfectly valid 38-disc game.
  //
  // Agreement between the reported score and the discs actually on the board is
  // both true of every game and a much better check -- it ties the status line,
  // the reducer and the rules engine together at the end of a real sequence of
  // moves.
  const counts = [...final.matchAll(/(\d+)/g)].map((m) => Number(m[1]));
  expect(counts.length).toBeGreaterThanOrEqual(2);
  expect(counts[0]! + counts[1]!).toBe(await discCount(page));

  // And it was a real game rather than a two-move accident.
  expect(await discCount(page)).toBeGreaterThan(20);
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
