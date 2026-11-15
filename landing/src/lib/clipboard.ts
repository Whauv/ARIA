export type CopyResult = "clipboard" | "fallback" | "failed";

export interface CopyDeps {
  /** Async Clipboard API writer, if available in a secure context. */
  writeText?: (text: string) => Promise<void>;
  /** Legacy synchronous copy (hidden textarea + execCommand). */
  legacyCopy?: (text: string) => boolean;
}

/** Minimal slice of `Document` used by the legacy fallback (keeps it testable). */
export interface LegacyCopyDocument {
  createElement(tag: "textarea"): HTMLTextAreaElement;
  body: { appendChild(node: Node): unknown; removeChild(node: Node): unknown } | null;
  execCommand?(command: string): boolean;
  activeElement?: Element | null;
}

/**
 * Copy text via an off-screen textarea and `document.execCommand("copy")`.
 * Used when the Clipboard API is missing (insecure context, file://, older
 * browsers) or rejects. Always removes the textarea and restores focus.
 */
export function legacyCopy(text: string, doc: LegacyCopyDocument): boolean {
  if (!doc.body || typeof doc.execCommand !== "function") return false;
  const previouslyFocused = doc.activeElement as HTMLElement | null | undefined;
  const area = doc.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.setAttribute("aria-hidden", "true");
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.left = "-9999px";
  area.style.opacity = "0";
  doc.body.appendChild(area);
  let ok = false;
  try {
    area.select();
    ok = doc.execCommand("copy");
  } catch {
    ok = false;
  } finally {
    doc.body.removeChild(area);
    if (previouslyFocused && typeof previouslyFocused.focus === "function") {
      previouslyFocused.focus();
    }
  }
  return ok;
}

/** Try the Clipboard API first, then the legacy path. Never throws. */
export async function copyText(text: string, deps: CopyDeps): Promise<CopyResult> {
  if (deps.writeText) {
    try {
      await deps.writeText(text);
      return "clipboard";
    } catch {
      // Permission denied or insecure context: fall through to legacy copy.
    }
  }
  if (deps.legacyCopy) {
    try {
      if (deps.legacyCopy(text)) return "fallback";
    } catch {
      // Ignore and report failure below.
    }
  }
  return "failed";
}

/** Wire real browser APIs into `copyText`. */
export function browserCopyDeps(): CopyDeps {
  const deps: CopyDeps = {};
  if (
    typeof navigator !== "undefined" &&
    typeof window !== "undefined" &&
    window.isSecureContext &&
    navigator.clipboard?.writeText
  ) {
    deps.writeText = (t) => navigator.clipboard.writeText(t);
  }
  if (typeof document !== "undefined") {
    deps.legacyCopy = (t) => legacyCopy(t, document);
  }
  return deps;
}
