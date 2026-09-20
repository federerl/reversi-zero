/**
 * The puzzle page.
 *
 * One question, asked sixty times: *you are winning — find a move that keeps
 * it.* Only the first move is graded. Playing the whole ending out would need
 * either a solver in the browser or a precomputed tree that grows past anything
 * worth shipping, and the first move is where the reported mistakes happen
 * anyway.
 *
 * Every verdict is a lookup in a table computed before the page was built, so
 * the answer arrives with no delay and no qualification. This is the only screen
 * in the project that states something without an interval attached, and it can,
 * because the number was proved rather than measured.
 *
 * Nothing is locked. A gate would be the obvious reading of a staged
 * progression and the wrong call for a page strangers land on: hiding stage 4
 * behind twelve solved puzzles mostly stops a visitor seeing the interesting
 * ones. The order is carried by arrangement and labelling instead of by refusing
 * input.
 */

import { useEffect, useReducer } from "react";

import { Board, squareName } from "../ui/Board";
import { Button } from "../../../shared/ui/Button";
import { Shell } from "../../../shared/ui/Shell";
import { StatusPill } from "../ui/Panel";
import { safeStorage } from "../../../shared/theme";
import { passAction, type Action } from "../engine/rules";
import { STAGES, puzzlesInStage, stageOf, type Puzzle } from "./data";
import { readSolved, writeSolved } from "./progress";
import {
  accepting,
  boardOf,
  currentPuzzle,
  grade,
  newSession,
  reduce,
  solved,
  solvedInStage,
  type Session,
  type SessionAction,
  type Verdict,
} from "../state/puzzle";

export function PuzzlesScreen() {
  const [session, dispatch] = useReducer(
    (state: Session, action: SessionAction) =>
      reduce(state, action, puzzlesInStage(state.stage)),
    undefined,
    () => newSession(STAGES[0]?.number ?? 1, readSolved(safeStorage())),
  );

  // Saved as it changes rather than on the way out: there is no reliable moment
  // to catch a tab closing, and the record is four hundred bytes.
  useEffect(() => {
    writeSolved(safeStorage(), session.solvedIds);
  }, [session.solvedIds]);

  const stage = stageOf(session.stage);
  const inStage = puzzlesInStage(session.stage);
  const puzzle = currentPuzzle(session, inStage);

  if (puzzle === null) {
    return (
      <Shell breadcrumb="Endgame puzzles">
        <p className="panel p-4 text-ink-2">This stage has no puzzles in it yet.</p>
      </Shell>
    );
  }

  const verdict = session.played === null ? null : grade(puzzle, session.played);
  const showingLine = session.line !== null;
  const line = session.line ?? 0;
  const lineDone = line >= puzzle.principalVariation.length;
  const status = describe(puzzle, verdict, showingLine, line);

  const lastMove: Action | null = showingLine
    ? (puzzle.principalVariation[line - 1] ?? null)
    : session.played;

  return (
    <Shell breadcrumb="Endgame puzzles" wide>
      <StagePicker session={session} dispatch={dispatch} />

      <div className="game-layout puzzles-layout">
        <div className="puzzle-prompt panel flex flex-col gap-3 p-4">
          <p className="text-sm text-muted">
            {stage.title} · puzzle {session.index + 1} of {inStage.length} · {puzzle.empties}{" "}
            empty squares
          </p>

          <StatusPill
            headline={status.headline}
            detail={status.detail}
            tone={status.tone}
            thinking={false}
          />
        </div>

        <div className="game-table">
          <Board
            state={boardOf(session, puzzle)}
            interactive={accepting(session)}
            lastMove={lastMove}
            onPlay={(action) => dispatch({ type: "play", action })}
          />
        </div>

        <aside className="puzzle-panel panel flex flex-col gap-5 p-4">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              {showingLine && !lineDone && (
                <Button variant="primary" onClick={() => dispatch({ type: "stepLine" })}>
                  Next move
                </Button>
              )}
              {showingLine && (
                <Button onClick={() => dispatch({ type: "hideLine" })}>Back to the puzzle</Button>
              )}
              {verdict !== null && !solved(verdict) && (
                <Button variant="primary" onClick={() => dispatch({ type: "retry" })}>
                  Try again
                </Button>
              )}
              {verdict !== null && !showingLine && (
                <Button onClick={() => dispatch({ type: "showLine" })}>Show the winning line</Button>
              )}
              {(verdict !== null || showingLine) && session.index < inStage.length - 1 && (
                <Button
                  variant={verdict !== null && solved(verdict) ? "primary" : "secondary"}
                  onClick={() => dispatch({ type: "next" })}
                >
                  Next puzzle
                </Button>
              )}
            </div>
          </div>

          <p className="text-[0.95rem] leading-relaxed text-muted">{stage.teaches}</p>

          <MoveTable puzzle={puzzle} played={session.played} />
        </aside>
      </div>
    </Shell>
  );
}

function StagePicker({
  session,
  dispatch,
}: {
  session: Session;
  dispatch: (action: SessionAction) => void;
}) {
  return (
    <nav aria-label="Stages" className="mb-4 flex flex-wrap gap-2">
      {STAGES.map((stage) => {
        const inStage = puzzlesInStage(stage.number);
        const done = solvedInStage(session, inStage);
        const here = stage.number === session.stage;
        return (
          <button
            key={stage.number}
            type="button"
            aria-current={here ? "true" : undefined}
            // The full name is on the button where there is room for it and in
            // the accessible name always, so a narrow screen loses the width
            // rather than the meaning. Five buttons carrying "Stage 3 · Two
            // regions at once" wrap to five lines on a phone and push the board
            // off the bottom of it.
            aria-label={`Stage ${stage.number}, ${stage.title}, ${done} of ${inStage.length} solved`}
            onClick={() => dispatch({ type: "pickStage", stage: stage.number })}
            className={`btn ${here ? "btn-primary" : ""}`}
          >
            <span aria-hidden="true">
              Stage {stage.number}
              <span className="hidden lg:inline"> · {stage.title}</span>
              {/* Dimmed by opacity rather than by the muted colour: the
                  selected stage sits on brass, where a grey meant for the page
                  background is close to unreadable. */}
              <span className="opacity-65">
                {" "}
                {done}/{inStage.length}
              </span>
            </span>
          </button>
        );
      })}
    </nav>
  );
}

/**
 * Every legal move and the score it really leads to.
 *
 * Behind a disclosure, following the pattern the level notes already use: the
 * page answers the question first, and the working is one click away for
 * somebody who wants it. It is also the part that makes the page teach rather
 * than test -- "wrong" is not a lesson, and a full ranking is.
 */
function MoveTable({ puzzle, played }: { puzzle: Puzzle; played: Action | null }) {
  const size = puzzle.state.size;
  const ranked = [...puzzle.margins].sort((a, b) => b[1] - a[1] || a[0] - b[0]);

  return (
    <details className="disclosure">
      <summary>Every move, and where it leads</summary>
      <div className="flex flex-col gap-3 pb-1 text-sm text-muted">
        <ul className="flex flex-col gap-1">
          {ranked.map(([move, margin]) => (
            <li key={move} className="flex items-baseline gap-2">
              <span className="w-8 shrink-0 font-medium text-ink-2">
                {squareName(move, size)}
              </span>
              <span className="w-24 shrink-0 tabular-nums">
                {margin > 0 ? `wins by ${margin}` : margin < 0 ? `loses by ${-margin}` : "draws"}
              </span>
              <span className="min-w-0">
                {[
                  margin === puzzle.best ? "best" : "",
                  move === puzzle.tempting ? "the move the network likes" : "",
                  move === played ? "you played this" : "",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </li>
          ))}
        </ul>
        <p className="leading-relaxed">
          These are not estimates. With {puzzle.empties} squares empty the position is small
          enough to search to the last move, so each number is the disc difference the game
          really ends on when both sides play perfectly.
          {puzzle.greedyFalls && " Here, taking the most discs available loses."}
        </p>
      </div>
    </details>
  );
}

function describe(
  puzzle: Puzzle,
  verdict: Verdict | null,
  showingLine: boolean,
  line: number,
): { headline: string; detail: string; tone: "you" | "quiet" } {
  const mover = puzzle.state.toMove === 0 ? "Black" : "White";

  if (showingLine) {
    const total = puzzle.principalVariation.length;
    const size = puzzle.state.size;
    const played = puzzle.principalVariation
      .slice(0, line)
      .map((action) => (action === passAction(size) ? "pass" : squareName(action, size)))
      .join(" ");
    return {
      headline: line === 0 ? "Perfect play from here" : played,
      detail:
        line >= total
          ? `That is the whole game. ${mover} wins by ${puzzle.best}.`
          : `${line} of ${total} moves. It ends ${puzzle.best > 0 ? "+" : ""}${puzzle.best} for ${mover.toLowerCase()}.`,
      tone: "quiet",
    };
  }

  if (verdict === null) {
    return {
      headline: `${mover} to play, and winning`,
      detail: `Find a move that keeps the win. ${puzzle.empties} squares left.`,
      tone: "you",
    };
  }

  const name = squareName(verdict.move, puzzle.state.size);
  if (verdict.outcome === "wins") {
    return verdict.optimal
      ? { headline: `${name} wins by ${verdict.margin}.`, detail: "Nothing does better.", tone: "you" }
      : {
          headline: `${name} wins by ${verdict.margin}.`,
          detail: `Still a win. The best move wins by ${verdict.best}.`,
          tone: "you",
        };
  }
  if (verdict.outcome === "draws") {
    return {
      headline: `${name} draws.`,
      detail: `It gives away a win of ${verdict.best}.`,
      tone: "quiet",
    };
  }
  return {
    headline: `${name} loses by ${-verdict.margin}.`,
    detail: `The best move wins by ${verdict.best}.`,
    tone: "quiet",
  };
}
