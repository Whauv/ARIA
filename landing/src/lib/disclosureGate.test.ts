import { describe, expect, it } from "vitest";
import { COLLAPSE_MAX_WIDTH, NARROW_QUERY, defaultGroupOpen } from "./disclosureGate";

describe("defaultGroupOpen", () => {
  it("keeps reference groups open on a roomy screen", () => {
    expect(defaultGroupOpen(false)).toBe(true);
  });

  it("collapses them on a narrow screen", () => {
    expect(defaultGroupOpen(true)).toBe(false);
  });
});

describe("NARROW_QUERY", () => {
  it("is a max-width query built from the documented breakpoint", () => {
    expect(NARROW_QUERY).toBe(`(max-width: ${COLLAPSE_MAX_WIDTH}px)`);
  });

  it("uses a breakpoint that treats phones as narrow and tablets as roomy", () => {
    expect(COLLAPSE_MAX_WIDTH).toBeGreaterThanOrEqual(390);
    expect(COLLAPSE_MAX_WIDTH).toBeLessThan(768);
  });
});
