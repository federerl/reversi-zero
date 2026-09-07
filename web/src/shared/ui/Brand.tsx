/**
 * The mark: two discs, one turning.
 *
 * A Reversi disc has two faces and the game is the act of turning it over, so
 * the emblem is that and nothing else -- a lacquer disc overlapping an ivory
 * one. Drawn inline rather than fetched, because it is nine lines of SVG and the
 * site loads no images.
 */

export function Emblem({ size = 22 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 32 24"
      width={(size * 32) / 24}
      height={size}
      aria-hidden="true"
      className="shrink-0"
    >
      <circle
        cx="11"
        cy="12"
        r="9.5"
        fill="var(--color-disc-black)"
        stroke="var(--color-ink)"
        strokeOpacity="0.42"
      />
      <circle
        cx="21"
        cy="12"
        r="9.5"
        fill="var(--color-disc-white)"
        stroke="var(--color-ground)"
        strokeOpacity="0.35"
      />
    </svg>
  );
}

/**
 * The site's name, as a link home unless this *is* home.
 *
 * "Reversi Zero" is the one all-caps element on the site. It is a name rather
 * than a label, which is the case where letter-spaced capitals earn their place.
 */
export function Wordmark({ as = "link", size = "md" }: { as?: "link" | "text"; size?: "md" | "lg" }) {
  const inner = (
    <>
      <Emblem size={size === "lg" ? 28 : 22} />
      <span className={`wordmark ${size === "lg" ? "text-2xl" : "text-lg"}`}>Reversi Zero</span>
    </>
  );
  if (as === "text") {
    return <p className="flex items-center gap-2.5 text-ink">{inner}</p>;
  }
  return (
    <a href="/" className="flex items-center gap-2.5 text-ink hover:opacity-80" aria-label="Reversi Zero, all games">
      {inner}
    </a>
  );
}
