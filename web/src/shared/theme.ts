/**
 * Dark or light. Dark is the site's default; light is a choice.
 *
 * The stylesheet reads one attribute, `data-theme` on the root element. Absent,
 * the page is dark; set to "light", the light palette applies. That keeps every
 * colour on the page flowing from the same tokens whichever way it was chosen.
 *
 * The choice is remembered in the browser only. A tiny inline script in each
 * page's `<head>` applies it before the first paint, so a visitor who chose
 * light does not see a dark flash on every load.
 */

export type Theme = "light" | "dark";
export type ThemeChoice = Theme | "default";

export const THEME_KEY = "rz:theme:v1";

/** What the page shows for a choice. The default is dark. */
export function resolveTheme(choice: ThemeChoice): Theme {
  return choice === "light" ? "light" : "dark";
}

/** The choice to make when the visitor asks for the other theme. */
export function toggledChoice(current: ThemeChoice): Theme {
  return resolveTheme(current) === "dark" ? "light" : "dark";
}

/** Read the stored choice. Anything unreadable or unknown is the default. */
export function readChoice(storage: Pick<Storage, "getItem"> | null | undefined): ThemeChoice {
  try {
    const raw = storage?.getItem(THEME_KEY);
    return raw === "light" || raw === "dark" ? raw : "default";
  } catch {
    return "default";
  }
}

/** Apply a choice to the document and remember it. Storage failures are ignored. */
export function applyChoice(
  choice: ThemeChoice,
  root: { dataset: DOMStringMap } = document.documentElement,
  storage: Pick<Storage, "setItem" | "removeItem"> | null | undefined = safeStorage(),
): void {
  if (choice === "default") delete root.dataset["theme"];
  else root.dataset["theme"] = choice;
  try {
    if (choice === "default") storage?.removeItem(THEME_KEY);
    else storage?.setItem(THEME_KEY, choice);
  } catch {
    // A private window or a browser that blocks storage: the theme still
    // applies for this page, it just is not remembered.
  }
}

export function safeStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
