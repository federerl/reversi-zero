/**
 * Light or dark, by choice or by the system's preference.
 *
 * The stylesheet reads one attribute, `data-theme` on the root element. Absent,
 * the system preference decides through `prefers-color-scheme`; present, it
 * wins. That keeps every colour on the page flowing from the same tokens
 * whichever way the theme was chosen.
 *
 * The choice is remembered in the browser only. A tiny inline script in each
 * page's `<head>` applies it before the first paint, so a visitor who chose dark
 * does not see a white flash on every load.
 */

export type Theme = "light" | "dark";
export type ThemeChoice = Theme | "system";

export const THEME_KEY = "rz:theme:v1";

/** What the page actually shows, given the choice and what the system prefers. */
export function resolveTheme(choice: ThemeChoice, prefersDark: boolean): Theme {
  if (choice === "system") return prefersDark ? "dark" : "light";
  return choice;
}

/** The choice to make when the visitor asks for the other theme. */
export function toggledChoice(current: ThemeChoice, prefersDark: boolean): Theme {
  return resolveTheme(current, prefersDark) === "dark" ? "light" : "dark";
}

/** Read the stored choice. Anything unreadable or unknown is "system". */
export function readChoice(storage: Pick<Storage, "getItem"> | null | undefined): ThemeChoice {
  try {
    const raw = storage?.getItem(THEME_KEY);
    return raw === "light" || raw === "dark" ? raw : "system";
  } catch {
    return "system";
  }
}

/** Apply a choice to the document and remember it. Storage failures are ignored. */
export function applyChoice(
  choice: ThemeChoice,
  root: { dataset: DOMStringMap } = document.documentElement,
  storage: Pick<Storage, "setItem" | "removeItem"> | null | undefined = safeStorage(),
): void {
  if (choice === "system") delete root.dataset["theme"];
  else root.dataset["theme"] = choice;
  try {
    if (choice === "system") storage?.removeItem(THEME_KEY);
    else storage?.setItem(THEME_KEY, choice);
  } catch {
    // A private window or a browser that blocks storage: the theme still
    // applies for this page, it just is not remembered.
  }
}

export function systemPrefersDark(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches;
}

export function safeStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
