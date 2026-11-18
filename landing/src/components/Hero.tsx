import { REPO_URL } from "../content";
import type { TextRun } from "../lib/textRuns";
import { HeroInstrument } from "./HeroInstrument";
import { Magnetic } from "./Magnetic";
import { TextEffect } from "./TextEffect";

/** The headline, in styled runs so the accent colours survive the word split. */
const TITLE: TextRun[] = [
  { text: "Draw", className: "accent-draw" },
  { text: "in the air." },
  { text: "Pick it up.", className: "accent-select" },
  { text: "Move it." },
];

export function Hero() {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="wrap">
        <div>
          <p className="eyebrow">Desktop app · Python · Webcam</p>
          <TextEffect as="h1" id="hero-title" runs={TITLE} delay={0.1} />
          <p className="lede">
            ARIA is a Python app that uses your webcam to track one hand: sketch with a fingertip, turn
            the drawing into a sprite, then pinch to select, drag, and resize it — with voice commands
            and a keyboard for everything else.
          </p>
          <div className="hero-ctas">
            <Magnetic>
              <a className="btn btn--primary" href="#run">
                Run it locally
              </a>
            </Magnetic>
            <Magnetic>
              <a className="btn btn--ghost" href={REPO_URL} rel="noopener">
                View on GitHub
              </a>
            </Magnetic>
          </div>
          <ul className="hero-facts" aria-label="At a glance">
            <li>Hand tracking runs locally</li>
            <li>Undo, redo, save, export PNG</li>
            <li>MIT licensed</li>
          </ul>
        </div>
        <HeroInstrument />
      </div>
    </section>
  );
}
