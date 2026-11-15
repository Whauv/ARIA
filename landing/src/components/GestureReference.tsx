import { gestureGroups, keyboard } from "../content";

export function GestureReference() {
  return (
    <section className="section" id="gestures" aria-labelledby="gestures-title">
      <div className="wrap">
        <div className="section-head">
          <p className="eyebrow">Reference</p>
          <h2 id="gestures-title">Gestures, with keyboard equivalents.</h2>
          <p className="lede">
            One primary hand drives everything. The thresholds below are the ones the app uses. Every
            gesture action also has a keyboard or mouse path in the app window.
          </p>
        </div>

        <div className="ref">
          {gestureGroups.map((g) => (
            <div className="ref-group" key={g.id}>
              <h3 id={g.id}>
                <span className={`chip chip--${g.mode}`}>{g.mode === "any" ? "ANY" : g.mode.toUpperCase()}</span>
                {g.title}
              </h3>
              <div className="table-scroll">
                <table className="ref-table" aria-labelledby={g.id}>
                  <thead>
                    <tr>
                      <th scope="col">Gesture</th>
                      <th scope="col">Result</th>
                      <th scope="col">Keyboard / mouse</th>
                    </tr>
                  </thead>
                  <tbody>
                    {g.rows.map((r) => (
                      <tr key={r.gesture}>
                        <th scope="row">{r.gesture}</th>
                        <td>{r.result}</td>
                        <td className="keys">{r.keys ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {g.note && (
                <p className="ref-note">
                  <strong>Tracking lost:</strong> {g.note}
                </p>
              )}
            </div>
          ))}

          <div className="ref-group">
            <h3 id="keys-title">Keyboard &amp; mouse</h3>
            <dl className="keys-grid" aria-labelledby="keys-title">
              {keyboard.map((k) => (
                <div key={k.action}>
                  <dt>
                    {k.keys.map((key, i) => (
                      <span key={key}>
                        {i > 0 && <span className="or"> or </span>}
                        <kbd>{key}</kbd>
                      </span>
                    ))}
                  </dt>
                  <dd>{k.action}</dd>
                </div>
              ))}
            </dl>
            <p className="mouse-note">
              Mouse: click the toolbar, color swatches, and layer thumbnails. In Select mode, click a
              sprite to select it, drag to move it, and click empty space to deselect.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
