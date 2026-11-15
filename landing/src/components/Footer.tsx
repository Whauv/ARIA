import { LICENSE_URL, README_URL, REPO_URL } from "../content";

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <p>ARIA — a desktop Python app for drawing with gestures and voice.</p>
        <ul>
          <li>
            <a href={REPO_URL} rel="noopener">
              GitHub
            </a>
          </li>
          <li>
            <a href={README_URL} rel="noopener">
              README
            </a>
          </li>
          <li>
            <a href={LICENSE_URL} rel="noopener">
              MIT license
            </a>
          </li>
        </ul>
      </div>
    </footer>
  );
}
