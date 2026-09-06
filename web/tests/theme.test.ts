/**
 * The theme is one attribute and one stored word. These pin how the word is
 * read, what "the other theme" means, and that a broken storage never breaks
 * the page.
 */

import { describe, expect, it } from "vitest";

import { readMuted, writeMuted } from "../src/shared/sound";
import { THEME_KEY, applyChoice, readChoice, resolveTheme, toggledChoice } from "../src/shared/theme";

class FakeStorage {
  readonly data = new Map<string, string>();
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.data.set(key, value);
  }
  removeItem(key: string) {
    this.data.delete(key);
  }
}

class BrokenStorage {
  getItem(): string | null {
    throw new Error("blocked");
  }
  setItem(): void {
    throw new Error("blocked");
  }
  removeItem(): void {
    throw new Error("blocked");
  }
}

describe("resolving the theme", () => {
  it("is dark unless light was chosen", () => {
    expect(resolveTheme("default")).toBe("dark");
    expect(resolveTheme("dark")).toBe("dark");
    expect(resolveTheme("light")).toBe("light");
  });

  it("toggles to the opposite of what is showing", () => {
    expect(toggledChoice("default")).toBe("light");
    expect(toggledChoice("dark")).toBe("light");
    expect(toggledChoice("light")).toBe("dark");
  });
});

describe("remembering the theme", () => {
  it("reads only the two known words and treats anything else as the default", () => {
    const storage = new FakeStorage();
    expect(readChoice(storage)).toBe("default");
    storage.setItem(THEME_KEY, "light");
    expect(readChoice(storage)).toBe("light");
    storage.setItem(THEME_KEY, "purple");
    expect(readChoice(storage)).toBe("default");
    expect(readChoice(null)).toBe("default");
  });

  it("applies the attribute and stores the choice, and the default clears both", () => {
    const storage = new FakeStorage();
    const root = { dataset: {} as DOMStringMap };
    applyChoice("light", root, storage);
    expect(root.dataset["theme"]).toBe("light");
    expect(storage.getItem(THEME_KEY)).toBe("light");
    applyChoice("default", root, storage);
    expect(root.dataset["theme"]).toBeUndefined();
    expect(storage.getItem(THEME_KEY)).toBeNull();
  });

  it("still applies the attribute when storage throws", () => {
    const root = { dataset: {} as DOMStringMap };
    expect(() => applyChoice("light", root, new BrokenStorage())).not.toThrow();
    expect(root.dataset["theme"]).toBe("light");
    expect(readChoice(new BrokenStorage())).toBe("default");
  });
});

describe("remembering mute", () => {
  it("round-trips and defaults to sound on", () => {
    const storage = new FakeStorage();
    expect(readMuted(storage)).toBe(false);
    writeMuted(true, storage);
    expect(readMuted(storage)).toBe(true);
    writeMuted(false, storage);
    expect(readMuted(storage)).toBe(false);
    expect(readMuted(new BrokenStorage())).toBe(false);
    expect(() => writeMuted(true, new BrokenStorage())).not.toThrow();
  });
});
