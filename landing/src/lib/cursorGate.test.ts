import { describe, expect, it } from "vitest";
import {
  EDITABLE_SELECTOR,
  INTERACTIVE_SELECTOR,
  classifyCursorTarget,
  shouldEnableCursor,
} from "./cursorGate";

describe("shouldEnableCursor", () => {
  it("enables only for fine hover pointers without reduced motion", () => {
    expect(shouldEnableCursor({ finePointer: true, reducedMotion: false })).toBe(true);
    expect(shouldEnableCursor({ finePointer: true, reducedMotion: true })).toBe(false);
    expect(shouldEnableCursor({ finePointer: false, reducedMotion: false })).toBe(false);
    expect(shouldEnableCursor({ finePointer: false, reducedMotion: true })).toBe(false);
  });
});

/** Fake element whose `closest` matches if any comma-separated selector part is listed. */
function el(...matches: string[]) {
  return {
    closest(selector: string) {
      const parts = selector.split(",").map((s) => s.trim());
      return parts.some((p) => matches.includes(p)) ? {} : null;
    },
  };
}

describe("classifyCursorTarget", () => {
  it("hides over editable controls", () => {
    expect(classifyCursorTarget(el("input"))).toBe("hidden");
    expect(classifyCursorTarget(el("textarea"))).toBe("hidden");
    expect(classifyCursorTarget(el("select"))).toBe("hidden");
    expect(classifyCursorTarget(el('[contenteditable="true"]'))).toBe("hidden");
  });

  it("grows over links and buttons", () => {
    expect(classifyCursorTarget(el("a[href]"))).toBe("interactive");
    expect(classifyCursorTarget(el("button"))).toBe("interactive");
  });

  it("prefers hidden when an editable sits inside an interactive ancestor", () => {
    expect(classifyCursorTarget(el("label[for]", "input"))).toBe("hidden");
  });

  it("treats plain elements and non-elements (text nodes, null) as default", () => {
    expect(classifyCursorTarget(el())).toBe("default");
    expect(classifyCursorTarget(null)).toBe("default");
    expect(classifyCursorTarget({ nodeType: 3 })).toBe("default");
  });

  it("does not treat anchors without href as interactive", () => {
    expect(INTERACTIVE_SELECTOR).toContain("a[href]");
    expect(INTERACTIVE_SELECTOR.split(",").map((s) => s.trim())).not.toContain("a");
    expect(EDITABLE_SELECTOR).not.toContain('contenteditable="false"');
  });
});
