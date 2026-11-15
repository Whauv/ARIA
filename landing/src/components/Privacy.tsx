export function Privacy() {
  return (
    <section className="section" id="privacy" aria-labelledby="privacy-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Privacy</p>
          <h2 id="privacy-title">What stays local, and what can leave.</h2>
          <p className="lede">
            ARIA runs on your computer. Two optional features use Google services; everything else
            happens on your machine.
          </p>
        </div>

        <div className="privacy">
          <div>
            <h3>
              <span className="chip chip--local">Local</span>
              On your machine
            </h3>
            <ul>
              <li><strong>Camera capture</strong> and <strong>MediaPipe hand tracking</strong></li>
              <li>Drawing, sprites, undo/redo</li>
              <li>Project save/load and PNG export</li>
              <li>
                <strong>Wake-word detection</strong> with openWakeWord (default phrase “hey jarvis”)
              </li>
              <li>Text-to-speech replies with pyttsx3 (offline)</li>
            </ul>
          </div>
          <div>
            <h3>
              <span className="chip chip--cloud">Optional cloud</span>
              Sent to Google
            </h3>
            <ul>
              <li>
                <strong>Voice command transcription</strong>, after the wake word, via Google's web
                speech recognition (SpeechRecognition library). On by default; disable with{" "}
                <code>ALLOW_CLOUD_SPEECH_RECOGNITION=0</code>.
              </li>
              <li>
                <strong>“What did I draw”</strong> sends one current frame — camera plus drawing,
                without UI chrome — to Google Gemini, and only if you set an API key.
              </li>
            </ul>
          </div>
        </div>
        <p className="privacy-foot">
          Nothing else leaves your machine. No telemetry. No recordings are saved.
        </p>
      </div>
    </section>
  );
}
