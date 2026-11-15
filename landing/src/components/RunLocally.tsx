import { README_URL, envExample, envVars, setupSteps } from "../content";
import { CodeBlock } from "./CodeBlock";

export function RunLocally() {
  return (
    <section className="section" id="run" aria-labelledby="run-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Run locally</p>
          <h2 id="run-title">From clone to running in a few commands.</h2>
          <p className="lede">
            ARIA needs Python 3.12, a webcam, and optionally a microphone. Run these from a terminal.
          </p>
        </div>

        <div className="run">
          <div className="run-main">
            <ol className="setup-steps">
              {setupSteps.map((s) => (
                <li key={s.label}>
                  <p className="setup-label">{s.label}</p>
                  <CodeBlock code={s.code} label={s.label.toLowerCase()} />
                </li>
              ))}
            </ol>
            <p className="run-after">
              In the app window, Help (H) opens the in-app help and <kbd>S</kbd> shows the status
              panel. Full details are in the{" "}
              <a href={README_URL} rel="noopener">
                README
              </a>
              .
            </p>
          </div>

          <aside aria-labelledby="env-title">
            <h3 id="env-title">Optional .env</h3>
            <p>
              Create a <code>.env</code> file in the project folder to enable or adjust optional
              features. Everything works without it except Gemini scene description.
            </p>
            <dl className="env-table">
              {envVars.map((v) => (
                <div key={v.name}>
                  <dt>{v.name}</dt>
                  <dd>{v.description}</dd>
                </div>
              ))}
            </dl>
            <div style={{ marginTop: "1.25rem" }}>
              <CodeBlock code={envExample} label="example .env file" />
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}
