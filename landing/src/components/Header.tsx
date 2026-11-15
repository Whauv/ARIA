import { REPO_URL } from "../content";

const links = [
  { href: "#how-it-works", label: "How it works" },
  { href: "#gestures", label: "Gestures" },
  { href: "#voice", label: "Voice" },
  { href: "#privacy", label: "Privacy" },
  { href: "#run", label: "Run locally" },
];

export function Header() {
  return (
    <header className="site-header">
      <div className="wrap">
        <a className="wordmark" href="#top" aria-label="ARIA — back to top">
          <span className="wordmark-mark" aria-hidden="true" />
          ARIA
        </a>
        <nav className="site-nav" aria-label="Primary">
          <ul>
            {links.map((l) => (
              <li key={l.href}>
                <a href={l.href}>{l.label}</a>
              </li>
            ))}
            <li>
              <a className="nav-ext" href={REPO_URL} rel="noopener">
                GitHub
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}
