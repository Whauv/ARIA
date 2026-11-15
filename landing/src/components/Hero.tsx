import { REPO_URL } from "../content";
import { HeroInstrument } from "./HeroInstrument";

export function Hero() {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="wrap">
        <div>
          <p className="eyebrow">Desktop app · Python · Webcam</p>
          <h1 id="hero-title">
            <span className="accent-draw">Draw</span> in the air.{" "}
            <span className="accent-select">Pick it up.</span> Move it.
          </h1>
          <p className="lede">
            ARIA is a Python app that uses your webcam to track one hand: sketch with a fingertip, turn
            the drawing into a sprite, then pinch to select, drag, and resize it — with voice commands
            and a keyboard for everything else.
          </p>
          <div className="hero-ctas">
            <a className="btn btn--primary" href="#run">
              Run it locally
            </a>
            <a className="btn btn--ghost" href={REPO_URL} rel="noopener">
              View on GitHub
            </a>
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
