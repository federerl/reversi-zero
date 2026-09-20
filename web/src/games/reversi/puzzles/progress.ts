/**
 * Which puzzles somebody has already worked out, remembered in their browser.
 *
 * A bookmark, not a claim. The roadmap cut records and streaks because a tally
 * kept in a visitor's own browser can never be verified and so cannot be
 * evidence of anything. That objection does not apply to remembering which
 * puzzles a person has already seen, which is ordinary courtesy: reopening the
 * page and being handed puzzle one again is the annoying part.
 *
 * Every read is defensive, and deliberately so. A private window, blocked
 * cookies, cleared site data, a half-written value or a key somebody typed by
 * hand all have to end in "nothing solved yet" rather than a blank page. The
 * theme store already holds this property; so does this.
 *
 * Ids are positions, so a regenerated curriculum does the right thing without
 * being told: puzzles that survive keep their mark, and ones that do not simply
 * stop being listed.
 */

const KEY = "rz:puzzles:v1";

/**
 * Versioned in the key rather than inside the value. Bumping `v1` retires the
 * old record outright, which is what a curriculum change wants -- there is no
 * sensible way to migrate "solved" across a different set of positions, and
 * pretending otherwise would credit somebody for a puzzle they never saw.
 */
export const PROGRESS_KEY = KEY;

type Reader = Pick<Storage, "getItem"> | null | undefined;
type Writer = Pick<Storage, "setItem"> | null | undefined;

export function readSolved(storage: Reader): Set<string> {
  try {
    const raw = storage?.getItem(KEY);
    if (raw === null || raw === undefined) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((id): id is string => typeof id === "string"));
  } catch {
    // Unreadable storage, or a value that is not JSON. Either way the honest
    // answer is that nothing is known to be solved.
    return new Set();
  }
}

export function writeSolved(storage: Writer, solved: ReadonlySet<string>): void {
  try {
    // Sorted, so the stored value does not churn on every save purely because a
    // Set iterates in insertion order.
    storage?.setItem(KEY, JSON.stringify([...solved].sort()));
  } catch {
    // Storage is full or blocked. The session still works; it just is not
    // remembered, which is strictly better than failing the page over it.
  }
}
