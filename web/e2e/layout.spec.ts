/**
 * The board is square, the page does not scroll sideways, and nothing important
 * is off screen. At a laptop, a small laptop, and a phone.
 *
 * These are the failures a screenshot review misses and a player notices
 * immediately. The squareness check is the important one: the grid's own box and
 * all sixty-four squares are measured, because a board that is a few pixels
 * taller than it is wide is a board with rectangular squares, and the fix for it
 * has been regressed once already.
 */

import { expect, test, type Page } from "@playwright/test";

const SIZES = [
  { name: "laptop", width: 1440, height: 900 },
  { name: "small laptop", width: 1024, height: 768 },
  { name: "phone", width: 390, height: 844 },
] as const;

async function boardIsSquare(page: Page) {
  const grid = page.getByRole("grid");
  const box = (await grid.boundingBox())!;
  // Within a pixel: sub-pixel layout is normal, a rectangle is not.
  expect(Math.abs(box.width - box.height)).toBeLessThan(1.5);

  const squares = await page.locator("[data-square]").evaluateAll((nodes) =>
    nodes.map((node) => {
      const rect = node.getBoundingClientRect();
      return { w: rect.width, h: rect.height };
    }),
  );
  expect(squares).toHaveLength(64);
  const widths = squares.map((s) => s.w);
  const heights = squares.map((s) => s.h);
  expect(Math.max(...widths) - Math.min(...widths)).toBeLessThan(1.5);
  expect(Math.max(...heights) - Math.min(...heights)).toBeLessThan(1.5);
  // And a square square: every cell as tall as it is wide.
  for (const square of squares) expect(Math.abs(square.w - square.h)).toBeLessThan(1.5);
}

async function noSidewaysScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

for (const size of SIZES) {
  test.describe(`at ${size.name}, ${size.width} by ${size.height}`, () => {
    test.use({ viewport: { width: size.width, height: size.height } });

    test("the launcher fits and its board is square", async ({ page }) => {
      await page.goto("/");
      await expect(page.getByRole("link", { name: "Play Othello" })).toBeVisible();
      await boardIsSquare(page);
      await noSidewaysScroll(page);
    });

    test("the game fits, its board is square, and the controls are reachable", async ({ page }) => {
      await page.goto("/reversi/");
      await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 60_000 });

      await boardIsSquare(page);
      await noSidewaysScroll(page);

      // Everything a player acts on is on the page and hittable. A 44 pixel
      // target is the smallest a thumb finds reliably.
      for (const name of ["New game", "Swap sides", "Take back"]) {
        const button = page.getByRole("button", { name });
        await expect(button).toBeVisible();
        const box = (await button.boundingBox())!;
        expect(box.height).toBeGreaterThanOrEqual(43);
      }
      const select = page.getByLabel("Level");
      await expect(select).toBeVisible();
      expect((await select.boundingBox())!.height).toBeGreaterThanOrEqual(43);
    });
  });
}

test.describe("on a laptop", () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test("the whole game is on screen without scrolling", async ({ page }) => {
    await page.goto("/reversi/");
    await expect(page.getByRole("status")).toContainText("Your turn", { timeout: 60_000 });

    const overflow = await page.evaluate(
      () => document.documentElement.scrollHeight - window.innerHeight,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
