import { voiceCommands } from "../content";

export function Voice() {
  return (
    <section className="section" id="voice" aria-labelledby="voice-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Voice</p>
          <h2 id="voice-title">Say the wake phrase, then the command.</h2>
          <p className="lede">
            Voice is optional and needs a microphone. Hands stay on the drawing; your voice handles
            the housekeeping.
          </p>
        </div>

        <div className="voice">
          <div className="voice-flow ticked">
            <ol>
              <li>
                <span>01</span>
                <strong>Say “hey jarvis”</strong>
                <p>The wake word is detected on your machine with openWakeWord. Nothing is sent to wake it.</p>
              </li>
              <li>
                <span>02</span>
                <strong>Say a command</strong>
                <p>
                  The command is transcribed with Google's web speech recognition (cloud). Turn this off with{" "}
                  <code>ALLOW_CLOUD_SPEECH_RECOGNITION=0</code> — the wake word still works, but spoken
                  commands can't be transcribed.
                </p>
              </li>
              <li>
                <span>03</span>
                <strong>ARIA acts and replies</strong>
                <p>Replies are spoken offline with pyttsx3, and a short notification confirms the action.</p>
              </li>
            </ol>
          </div>

          <ul className="voice-list" aria-label="Voice commands">
            {voiceCommands.map((c) => (
              <li key={c.phrases.join("|")}>
                <span className="phrases">
                  {c.phrases.map((p, i) => (
                    <span key={p}>
                      {i > 0 && <span className="or">or </span>}
                      <span className="phrase">{p}</span>
                    </span>
                  ))}
                </span>
                <span className="effect">
                  {c.effect}
                  {c.cloud && <span className="tag-cloud">GEMINI</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
