/** Default open/closed state for the collapsible reference groups.
 *
 * DOM-free so it can be unit-tested under the node test environment.
 */

/** Below this width the reference tables are tall enough to bury the page. */
export const COLLAPSE_MAX_WIDTH = 640;

export const NARROW_QUERY = `(max-width: ${COLLAPSE_MAX_WIDTH}px)`;

/**
 * Reference groups start open on a roomy screen and closed on a narrow one,
 * where three three-column tables would otherwise push everything below them
 * far out of reach. The reader can still open or close any group by hand.
 */
export function defaultGroupOpen(narrow: boolean): boolean {
  return !narrow;
}
