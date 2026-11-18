import { describe, expect, it } from "vitest";
import { magneticOffset } from "./magnetic";

/** A 100x40 button whose centre sits at (150, 120). */
const box = { left: 100, top: 100, width: 100, height: 40 };

describe("magneticOffset", () => {
  it("does not move the element when the pointer is at its centre", () => {
    expect(magneticOffset({ x: 150, y: 120 }, box, 0.32, 90)).toEqual({ x: 0, y: 0 });
  });

  it("pulls toward the pointer, never further than the pointer itself", () => {
    const offset = magneticOffset({ x: 170, y: 120 }, box, 0.32, 90);
    expect(offset.x).toBeGreaterThan(0);
    expect(offset.x).toBeLessThan(20);
    expect(offset.y).toBeCloseTo(0);
  });

  it("pulls the opposite way on the other side", () => {
    const right = magneticOffset({ x: 170, y: 120 }, box, 0.32, 90);
    const left = magneticOffset({ x: 130, y: 120 }, box, 0.32, 90);
    expect(left.x).toBeCloseTo(-right.x);
  });

  it("fades to nothing at the edge of range", () => {
    const edge = magneticOffset({ x: 150 + 90, y: 120 }, box, 0.32, 90);
    expect(edge).toEqual({ x: 0, y: 0 });
  });

  it("ignores a pointer beyond range", () => {
    expect(magneticOffset({ x: 1000, y: 1000 }, box, 0.32, 90)).toEqual({ x: 0, y: 0 });
  });

  it("weakens as the pointer moves away", () => {
    const near = magneticOffset({ x: 160, y: 120 }, box, 0.32, 90);
    const far = magneticOffset({ x: 200, y: 120 }, box, 0.32, 90);
    expect(far.x).toBeLessThan(near.x * 4);
  });

  it("stays still for a zero or negative intensity or range", () => {
    expect(magneticOffset({ x: 170, y: 120 }, box, 0, 90)).toEqual({ x: 0, y: 0 });
    expect(magneticOffset({ x: 170, y: 120 }, box, 0.32, 0)).toEqual({ x: 0, y: 0 });
  });
});
