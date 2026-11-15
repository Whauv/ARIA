import type { ReactNode } from "react";
import { DrawDiagram, SelectDiagram, SpriteDiagram, VoiceDiagram } from "./Diagrams";
import { Reveal } from "./Reveal";

interface Step {
  title: string;
  mode: "draw" | "select" | "any";
  body: ReactNode;
  spec: [string, ReactNode][];
  figure: ReactNode;
}

const steps: Step[] = [
  {
    title: "Draw",
    mode: "draw",
    body: "In Draw mode, raise your index finger and move it — the fingertip leaves a stroke on the canvas over your camera view. Raise your middle finger too and the pen lifts, so you can reposition without drawing.",
    spec: [
      ["Draw", "Index finger up"],
      ["Lift pen", "Index + middle finger up"],
      ["Quick color", "Pinch and hold 0.6 s away from controls (green ↔ white)"],
      ["Keys", <><kbd>D</kbd> Draw mode · <kbd>1</kbd>–<kbd>8</kbd> colors</>],
    ],
    figure: <DrawDiagram />,
  },
  {
    title: "Make a sprite",
    mode: "draw",
    body: "When the drawing is ready, close your hand into a fist and hold it. After 1.5 seconds the drawing becomes a sprite — an object you can select, move, resize, and delete. Undo brings back what you change.",
    spec: [
      ["Gesture", "Hold a closed fist for 1.5 s"],
      ["Undo / redo", <><kbd>U</kbd> / <kbd>R</kbd></>],
    ],
    figure: <SpriteDiagram />,
  },
  {
    title: "Select & arrange",
    mode: "select",
    body: "Switch to Select mode by dwelling on the toolbar button (hover your fingertip and hold until the ring fills) or press V. Pinch a sprite to select it, keep pinching to drag it, and release to place it.",
    spec: [
      ["Dwell", "0.8 s in Draw mode · 0.55 s in Select mode"],
      ["Select", "Pinch a sprite (topmost wins)"],
      ["Drag", "Keep pinching ~0.2 s, release to place"],
      ["Resize", "Both hands pinching ~0.3 s"],
      ["Delete", "Two quick pinches arm it, a third within 1.5 s deletes"],
    ],
    figure: <SelectDiagram />,
  },
  {
    title: "Ask with voice",
    mode: "any",
    body: "Say the wake phrase “hey jarvis”, then a command: change color, clear, undo, save, switch mode — or ask “what did I draw” to have Gemini describe the current frame when an API key is set.",
    spec: [
      ["Wake word", "Detected locally (openWakeWord)"],
      ["Commands", "Transcribed by Google's web speech recognition — can be disabled"],
      ["Replies", "Spoken offline (pyttsx3)"],
    ],
    figure: <VoiceDiagram />,
  },
];

const chipText = { draw: "Draw mode", select: "Select mode", any: "Any mode" } as const;

export function Walkthrough() {
  return (
    <section className="section" id="how-it-works" aria-labelledby="how-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">How it works</p>
          <h2 id="how-title">Four moves, one hand.</h2>
          <p className="lede">
            ARIA tracks one primary hand with MediaPipe. Gestures are deliberate: holds have fixed
            thresholds, and if tracking is lost, pending holds and drags are cancelled — nothing fires.
          </p>
        </div>
        <ol className="steps">
          {steps.map((s, i) => (
            <Reveal as="li" className="step" key={s.title}>
              <div>
                <p className="step-index">
                  {String(i + 1).padStart(2, "0")} / 04{" "}
                  <span className={`chip chip--${s.mode}`} style={{ marginLeft: "0.5rem" }}>
                    {chipText[s.mode]}
                  </span>
                </p>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
                <ul className="step-spec">
                  {s.spec.map(([k, v]) => (
                    <li key={k}>
                      <span className="k">{k}</span>
                      <span>{v}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="step-figure">{s.figure}</div>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
