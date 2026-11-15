/** Media query for devices with a precise pointer that can hover (mouse/trackpad). */
export const FINE_POINTER_QUERY = "(hover: hover) and (pointer: fine)";
export const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

export interface CursorCapabilities {
  finePointer: boolean;
  reducedMotion: boolean;
}

/** The accent cursor only runs on hover-capable fine pointers without reduced motion. */
export function shouldEnableCursor({ finePointer, reducedMotion }: CursorCapabilities): boolean {
  return finePointer && !reducedMotion;
}

export type CursorTarget = "hidden" | "interactive" | "default";

/** Elements where the accent ring steps aside (text entry keeps the native caret clear). */
export const EDITABLE_SELECTOR =
  'input, textarea, select, [contenteditable=""], [contenteditable="true"]';
/** Elements where the ring grows slightly to acknowledge an affordance. */
export const INTERACTIVE_SELECTOR = 'a[href], button, [role="button"], summary, label[for]';

interface ClosestCapable {
  closest(selector: string): unknown;
}

/** Classify the element under the pointer. Editable wins over interactive. */
export function classifyCursorTarget(target: unknown): CursorTarget {
  if (!target || typeof (target as ClosestCapable).closest !== "function") return "default";
  const el = target as ClosestCapable;
  if (el.closest(EDITABLE_SELECTOR)) return "hidden";
  if (el.closest(INTERACTIVE_SELECTOR)) return "interactive";
  return "default";
}
