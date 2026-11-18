import { AnimatedGroup, AnimatedItem } from "./AnimatedGroup";

const limits: [string, string][] = [
  ["Needs a webcam", "ARIA reads your hand from the camera feed; without one there is nothing to track."],
  [
    "Lighting and framing matter",
    "Tracking works best with even lighting and one hand in frame. Two-hand resize is the only gesture that uses both.",
  ],
  [
    "Voice needs a microphone and the cloud",
    "The wake word is local, but commands are transcribed by Google's web speech recognition. With that disabled, spoken commands can't be understood.",
  ],
  ["Scene description needs a key", "“What did I draw” requires a Google Gemini API key. Without one, AI stays off and the app says so."],
  ["Desktop only", "ARIA is a desktop Python app. It is not a web app or a mobile app; this page is just its documentation."],
  [
    "Accessibility",
    "Keyboard and mouse alternatives exist for gesture actions, but ARIA does not claim screen-reader accessibility for its OpenCV window.",
  ],
];

export function Limitations() {
  return (
    <section className="section" id="limitations" aria-labelledby="limits-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Limitations</p>
          <h2 id="limits-title">What to expect before you install.</h2>
        </div>
        <AnimatedGroup as="ul" className="limits">
          {limits.map(([title, body], i) => (
            <AnimatedItem as="li" key={title}>
              <span className="mark" aria-hidden="true">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span>
                <strong>{title}</strong>
                {body}
              </span>
            </AnimatedItem>
          ))}
        </AnimatedGroup>
      </div>
    </section>
  );
}
