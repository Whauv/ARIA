/** Pull maths for the magnetic CTA wrapper.
 *
 * DOM-free so it can be unit-tested under the node test environment.
 */

export interface Point {
  x: number;
  y: number;
}

/** Just the parts of a DOMRect the offset needs. */
export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * How far to shift an element toward the pointer.
 *
 * The pull is strongest at the centre and fades linearly to nothing at
 * `range`, so the element settles back as the pointer leaves rather than
 * snapping. Beyond `range` there is no pull at all.
 */
export function magneticOffset(
  pointer: Point,
  box: Box,
  intensity: number,
  range: number,
): Point {
  if (range <= 0 || intensity <= 0) return { x: 0, y: 0 };
  const centreX = box.left + box.width / 2;
  const centreY = box.top + box.height / 2;
  const dx = pointer.x - centreX;
  const dy = pointer.y - centreY;
  const distance = Math.hypot(dx, dy);
  if (distance >= range) return { x: 0, y: 0 };
  const falloff = 1 - distance / range;
  return { x: dx * intensity * falloff, y: dy * intensity * falloff };
}
