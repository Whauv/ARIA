import { describe, expect, it } from "vitest";
import { runsToText, runsToWords, staggerForWordCount } from "./textRuns";

describe("runsToWords", () => {
  it("splits each run into words and carries its style", () => {
    expect(
      runsToWords([
        { text: "Draw", className: "accent-draw" },
        { text: "in the air." },
      ]),
    ).toEqual([
      { text: "Draw", className: "accent-draw" },
      { text: "in" },
      { text: "the" },
      { text: "air." },
    ]);
  });

  it("collapses irregular whitespace instead of emitting empty words", () => {
    expect(runsToWords([{ text: "  Move   it.  \n Now " }])).toEqual([
      { text: "Move" },
      { text: "it." },
      { text: "Now" },
    ]);
  });

  it("returns nothing for empty or whitespace-only runs", () => {
    expect(runsToWords([])).toEqual([]);
    expect(runsToWords([{ text: "   " }])).toEqual([]);
  });

  it("omits className entirely when a run has none", () => {
    const [word] = runsToWords([{ text: "plain" }]);
    expect(Object.hasOwn(word, "className")).toBe(false);
  });
});

describe("runsToText", () => {
  it("joins runs into the sentence a screen reader should hear", () => {
    expect(
      runsToText([
        { text: "Draw", className: "accent-draw" },
        { text: "in the air." },
        { text: "Pick it up.", className: "accent-select" },
      ]),
    ).toBe("Draw in the air. Pick it up.");
  });

  it("skips runs that are only whitespace", () => {
    expect(runsToText([{ text: "One" }, { text: "  " }, { text: "two" }])).toBe("One two");
  });
});

describe("staggerForWordCount", () => {
  it("uses the base delay for short headlines", () => {
    expect(staggerForWordCount(5)).toBeCloseTo(0.055);
  });

  it("compresses so a long headline still finishes promptly", () => {
    const stagger = staggerForWordCount(40);
    expect(stagger).toBeLessThan(0.055);
    expect(stagger * 40).toBeCloseTo(0.6);
  });

  it("never returns zero or a negative delay", () => {
    expect(staggerForWordCount(0)).toBeGreaterThan(0);
    expect(staggerForWordCount(-3)).toBeGreaterThan(0);
  });
});
