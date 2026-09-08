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

  await page.goto("/reversi/");
  // Loading a network is a download plus a compile; the status line says so
  // until it is ready, and clicking before then would be testing nothing.
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 60_000 });
});

test("the opening position is the standard one", async ({ page }) => {
  expect(await discCount(page)).toBe(4);
  // d3, c4, f5, e6 -- the four openings available to black, every game.
  expect((await playableSquares(page)).sort()).toEqual(["19", "26", "37", "44"]);
});

test("the agent answers a move, and the game moves on", async ({ page }) => {
  await page.locator('[data-square="19"]').click();

  // It thinks, and then it does not.
  await expect(page.getByRole("status")).toContainText("thinking", { timeout: 5_000 });
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 30_000 });

  // Black played one disc and flipped one; white then played and flipped.
  expect(await discCount(page)).toBeGreaterThan(4);

  // And it reports what it did, rather than only that it did something.
  await expect(page.getByText(/Played \w+\d in \d+ ms/)).toBeVisible();
});

test("a network opponent searches, and says how much", async ({ page }) => {
  // The default level is a baseline, which runs no search. Picking a level a
  // network plays is what exercises the ONNX path -- and the simulation count in
  // the move report is the visible evidence that it ran.
  //
  // This is the only test that downloads and runs the network, so it is the only
  // one that can catch a broken ONNX path. On a slow shared runner that means a
  // download, a compile and a search, none of which are quick -- hence the
  // long waits and test.slow().
  test.slow();

  await page.getByLabel("Level").selectOption("gen05");
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 90_000 });

  // If the network failed to load, the app says so in a toast. Checking here
  // turns that into an immediate, readable failure instead of a later assertion
  // timing out for ninety seconds with no clue why.
  await expect(page.getByRole("alert")).toBeHidden();

  await expect(page.getByLabel("Thinking time")).toBeVisible();
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 90_000 });

  await expect(page.getByText(/Played \w+\d after \d+ simulations in \d+ ms/)).toBeVisible();
  // A network has an opinion about who is winning; the bar shows it.
  await expect(page.getByRole("meter")).toBeVisible();
});

test("a baseline level offers no search controls and claims no opinion", async ({ page }) => {
  // The lowest two levels pick from the rules alone. There is no simulation
  // budget to spend and no value to report, so the interface shows neither
  // rather than inventing them.
  await expect(page.getByLabel("Level")).toHaveValue("greedy");
  await expect(page.getByLabel("Thinking time")).toBeHidden();

  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 30_000 });

  await expect(page.getByRole("meter")).toBeHidden();
  await expect(page.getByText(/simulations/)).toBeHidden();
});

test("the board is never clickable while the agent is thinking", async ({ page }) => {
  // Otherwise a fast player can queue a move into a position that no longer
  // exists by the time it lands.
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("thinking");
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
    if (status.includes("thinking")) {
      await expect(page.getByRole("status")).not.toContainText("thinking", { timeout: 30_000 });
      continue;
    }

    // Decide from the controls, not from the prose.
    //
    // Two different states contain the words "must pass" -- "You have no legal
    // move. You must pass." and "The agent has no legal move and must pass." --
    // and only the first offers a button. Matching the phrase meant the test
    // occasionally tried to click a Pass button that was not there, and waited
    // out the whole test timeout when it did. Which of the two occurs depends on
    // the game, which is why it only failed sometimes.
    const pass = page.getByRole("button", { name: "Pass" });
    if (await pass.isVisible()) {
      await pass.click();
    } else {
      const squares = await playableSquares(page);
      if (squares.length === 0) {
        // Not our turn: the agent is moving, or passing on its own.
        await page.waitForTimeout(100);
        continue;
      }
      await page.locator(`[data-square="${squares[0]}"]`).click();
    }
    await expect(page.getByRole("status")).not.toContainText("thinking", { timeout: 30_000 });
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
  // The scoreline is matched by its shape, not by taking the first two numbers
  // in the sentence. The opponent is named by its rung -- "Level 2 wins, 47 to
  // 17" -- so the level number is a digit in the same string, and reading
  // digits positionally summed 2 + 47. It passed or failed depending on who won,
  // because "You win" contains no number.
  const margin = final.match(/(\d+) to (\d+)/);
  const drawn = final.match(/(\d+) discs each/);
  const reported = margin
    ? Number(margin[1]) + Number(margin[2])
    : 2 * Number(drawn![1]);
  expect(reported).toBe(await discCount(page));

  // And it was a real game rather than a two-move accident.
  expect(await discCount(page)).toBeGreaterThan(20);

  // The ending is a real ending: a dialog that names the result and offers the
  // next game, not a sentence in the status box. Its count agrees with the
  // status line and the board, and "Play again" starts over.
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  // The ending names the opponent it was played against, whichever it was.
  await expect(dialog.getByRole("heading", { level: 2 })).toHaveText(/You win|wins|A draw/);
  const dialogBlack = Number(await dialog.locator("[data-count='black']").textContent());
  const dialogWhite = Number(await dialog.locator("[data-count='white']").textContent());
  expect(dialogBlack + dialogWhite).toBe(reported);

  await dialog.getByRole("button", { name: "Rematch" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("status")).toContainText("Your turn");
  expect(await discCount(page)).toBe(4);
});

test("the theme and sound switches are on every page and remember their setting", async ({
  page,
}) => {
  const html = page.locator("html");
  await expect(html).not.toHaveAttribute("data-theme", /./);

  await page.getByRole("button", { name: /Switch to the (dark|light) theme/ }).click();
  await expect(html).toHaveAttribute("data-theme", /^(dark|light)$/);
  const chosen = await html.getAttribute("data-theme");

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", chosen!);

  const mute = page.getByRole("button", { name: "Mute sounds" });
  await mute.click();
  await expect(page.getByRole("button", { name: "Unmute sounds" })).toBeVisible();
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Unmute sounds" })).toBeVisible();
});

test("taking a move back returns the board to the player", async ({ page }) => {
  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 30_000 });
  const afterOneMove = await discCount(page);

  await page.getByRole("button", { name: "Take back" }).click();

  expect(await discCount(page)).toBeLessThan(afterOneMove);
  expect((await playableSquares(page)).sort()).toEqual(["19", "26", "37", "44"]);
});

test("a level is offered by number and word, with the measurement behind it", async ({
  page,
}) => {
  // A player picks a rung. What the rung *is* -- which checkpoint, what it
  // rates, how sure that rating is -- has to be reachable and has to be absent
  // until asked for, or the rule that no claim here goes unbacked would rest on
  // a JSON file nobody opens.
  const levels = page.getByLabel("Level");
  await expect(levels).toContainText(/Level 1, \w+/);
  await expect(levels).toContainText(/Level 6, \w+/);
  await expect(levels).not.toContainText(/Elo|Generation/);

  const about = page.locator("details", { hasText: /About level \d/ });
  await expect(about.getByText(/95% interval/)).toBeHidden();
  await about.locator("summary").click();
  await expect(about.getByText(/rates \+\d+ against the other levels/)).toBeVisible();
  await expect(about.getByText(/95% interval from \d+ to \d+/)).toBeVisible();
});

test("thinking time is offered as time, and rated where it is explained", async ({ page }) => {
  // Thinking time is the second dial, and it is the one whose promise is easy
  // to break: a time cap cannot advertise a simulation count it will not reach
  // on a slow device. So the option says how long it will take, and the rating
  // the calibration measured for it appears in the note beside the level.
  await page.getByLabel("Level").selectOption("gen05");
  const budgets = page.getByLabel("Thinking time");
  await expect(budgets).toBeVisible();

  await expect(budgets).toContainText(/Casual, \d+ simulations/);
  await expect(budgets).toContainText(/Max, up to \d\.\d s/);

  const about = page.locator("details", { hasText: /About level \d/ });
  await about.locator("summary").click();
  await expect(about.getByText(/rates \+\d+ in its own tournament/)).toBeVisible();
});

test("a played game can be reviewed move by move", async ({ page }) => {
  // The scrubber is absent until there is something to review, so the opening
  // position must not offer one.
  await expect(page.getByLabel("Move to review")).toBeHidden();

  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 30_000 });

  const scrubber = page.getByLabel("Move to review");
  await expect(scrubber).toBeVisible();

  const liveDiscs = await discCount(page);

  // Back to the opening position. Four discs, and the interface says plainly
  // that what is on screen is not the game.
  await page.getByRole("button", { name: "Previous" }).click();
  await page.getByRole("button", { name: "Previous" }).click();
  await expect(page.getByRole("status")).toContainText("Reviewing");
  expect(await discCount(page)).toBe(4);

  // And nothing is clickable while reviewing: a move played into a position
  // from three plies ago would land in a game that no longer exists.
  expect(await playableSquares(page)).toEqual([]);

  await page.getByRole("button", { name: "Latest" }).click();
  await expect(page.getByRole("status")).toContainText("Your turn");
  expect(await discCount(page)).toBe(liveDiscs);
});

test("reviewing while the AI thinks does not cancel its search", async ({ page }) => {
  // The reason the review is a cursor rather than a second copy of the game.
  // The screen restarts the agent's search whenever the position changes, so if
  // scrubbing changed the position the player would silently abort the move
  // they are waiting for -- and the symptom is a board that never answers.
  test.slow();

  await page.getByLabel("Level").selectOption("gen05");
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 90_000 });
  await page.getByLabel("Thinking time").selectOption("max");

  await page.locator('[data-square="19"]').click();
  await expect(page.getByRole("status")).toContainText("thinking");

  // Scrub back mid-search.
  //
  // This assertion races the reply, and deliberately: the AI's move clears the
  // review cursor, so "Reviewing" is only on screen until it lands. The margin
  // is the thinking time's two-second cap against one round-trip, which is wide
  // enough -- but a level whose cap dropped to a fraction of a second would make
  // this flaky rather than wrong, and that is the reason to look here first.
  await page.getByRole("button", { name: "Previous" }).click();
  await expect(page.getByRole("status")).toContainText("Reviewing");

  // The search was still running underneath, so the move arrives and the game
  // returns to the live position on its own.
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 90_000 });
  await expect(page.getByRole("alert")).toBeHidden();
  expect(await discCount(page)).toBeGreaterThan(4);
});

test("the board can be played with the keyboard alone", async ({ page }) => {
  await page.locator('[data-square="19"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 30_000 });
  expect(await discCount(page)).toBeGreaterThan(4);
});
