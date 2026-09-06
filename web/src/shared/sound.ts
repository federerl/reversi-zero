/**
 * The sounds of a board game, synthesised rather than downloaded.
 *
 * A disc going down is a short, dull knock; a disc turning over is a quieter,
 * higher tick; the end of a game is three notes. All of it is made with the Web
 * Audio API from oscillators and a burst of noise, so there are no files to
 * fetch and nothing to fail to load.
 *
 * Browsers refuse to start audio until the visitor has interacted with the
 * page, so the audio context is created lazily on the first sound after a
 * click, never at load. Mute is remembered in the browser only.
 */

import { safeStorage } from "./theme";

export const SOUND_KEY = "rz:sound:v1";

export function readMuted(storage: Pick<Storage, "getItem"> | null | undefined): boolean {
  try {
    return storage?.getItem(SOUND_KEY) === "muted";
  } catch {
    return false;
  }
}

export function writeMuted(
  muted: boolean,
  storage: Pick<Storage, "setItem" | "removeItem"> | null | undefined,
): void {
  try {
    if (muted) storage?.setItem(SOUND_KEY, "muted");
    else storage?.removeItem(SOUND_KEY);
  } catch {
    // Not remembered, still applied.
  }
}

class Sounds {
  private context: AudioContext | null = null;
  muted = readMuted(safeStorage());

  setMuted(muted: boolean): void {
    this.muted = muted;
    writeMuted(muted, safeStorage());
  }

  /** A disc placed on the board. */
  place(): void {
    this.play((ctx, at) => {
      this.knock(ctx, at, 140, 0.09, 0.5);
      this.noise(ctx, at, 0.035, 0.25);
    });
  }

  /** Discs turning over; `ring` staggers the ticks like the animation does. */
  flip(rings: number, stepMs: number): void {
    this.play((ctx, at) => {
      for (let ring = 1; ring <= Math.min(rings, 7); ring++) {
        this.knock(ctx, at + (ring * stepMs) / 1000, 420 + ring * 30, 0.05, 0.16);
      }
    });
  }

  /** The end of a game. Three rising notes for a win, falling for a loss, one for a draw. */
  gameOver(result: "win" | "loss" | "draw"): void {
    this.play((ctx, at) => {
      const notes =
        result === "win" ? [392, 494, 587] : result === "loss" ? [392, 349, 294] : [440];
      notes.forEach((frequency, i) => this.tone(ctx, at + i * 0.16, frequency, 0.32, 0.22));
    });
  }

  // -----------------------------------------------------------------------

  private play(schedule: (ctx: AudioContext, at: number) => void): void {
    if (this.muted) return;
    const ctx = this.ensureContext();
    if (ctx === null) return;
    if (ctx.state === "suspended") void ctx.resume();
    schedule(ctx, ctx.currentTime + 0.005);
  }

  private ensureContext(): AudioContext | null {
    if (this.context !== null) return this.context;
    try {
      this.context = new AudioContext();
    } catch {
      this.context = null;
    }
    return this.context;
  }

  /** A short, damped sine: the body of a knock or a tick. */
  private knock(ctx: AudioContext, at: number, frequency: number, seconds: number, gain: number) {
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(frequency, at);
    osc.frequency.exponentialRampToValueAtTime(Math.max(40, frequency * 0.6), at + seconds);
    amp.gain.setValueAtTime(gain, at);
    amp.gain.exponentialRampToValueAtTime(0.001, at + seconds);
    osc.connect(amp).connect(ctx.destination);
    osc.start(at);
    osc.stop(at + seconds + 0.02);
  }

  /** A burst of filtered noise: the click of disc on wood. */
  private noise(ctx: AudioContext, at: number, seconds: number, gain: number) {
    const length = Math.ceil(ctx.sampleRate * seconds);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / length);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.value = 1800;
    filter.Q.value = 0.8;
    const amp = ctx.createGain();
    amp.gain.value = gain;
    source.connect(filter).connect(amp).connect(ctx.destination);
    source.start(at);
  }

  /** A soft note with a slow release, for the ending. */
  private tone(ctx: AudioContext, at: number, frequency: number, seconds: number, gain: number) {
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.value = frequency;
    amp.gain.setValueAtTime(0.0001, at);
    amp.gain.exponentialRampToValueAtTime(gain, at + 0.02);
    amp.gain.exponentialRampToValueAtTime(0.001, at + seconds);
    osc.connect(amp).connect(ctx.destination);
    osc.start(at);
    osc.stop(at + seconds + 0.05);
  }
}

/** One instance for the page; a second would open a second audio context. */
export const sounds = new Sounds();
