/**
 * The position the launcher shows.
 *
 * A real one. It is replayed, move by move, from the first recorded game in the
 * cross-language fixtures -- a game the agent actually played against itself --
 * so the board on the front page is a position that occurred rather than an
 * arrangement chosen to look good. `apply` refuses an illegal move, so if this
 * ever stops being a legal game the page fails loudly instead of quietly
 * showing something impossible.
 */

import games from "../games/reversi/engine/__fixtures__/games.json";
import { apply, initialState, type State } from "../games/reversi/engine/rules";

/** Far enough in that both colours hold ground, before the board crowds. */
export const PREVIEW_PLIES = 24;

export function previewPosition(plies: number = PREVIEW_PLIES): State {
  const moves = (games as { games: Array<{ moves: number[] }> }).games[0]?.moves ?? [];
  let state = initialState(8);
  for (const move of moves.slice(0, plies)) {
    state = apply(state, move);
  }
  return state;
}
