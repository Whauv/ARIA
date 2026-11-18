/** Splitting helpers for the staggered text reveal.
 *
 * Kept free of React and the DOM so the behaviour is unit-testable under the
 * node test environment, the same way `cursorGate` backs `AccentCursor`.
 */

/** A stretch of the headline that shares one style (e.g. the accent colour). */
export interface TextRun {
  text: string;
  className?: string;
}

/** One animated segment: a word plus the style of the run it came from. */
export interface TextWord {
  text: string;
  className?: string;
}

/** The full sentence, for the screen-reader copy and the document title. */
export function runsToText(runs: readonly TextRun[]): string {
  return runs
    .map((run) => run.text.trim())
    .filter(Boolean)
    .join(" ");
}

/** Flatten runs into words, carrying each run's style onto its words.
 *
 * Whitespace is dropped rather than kept as its own segment: the words render
 * as inline-blocks separated by a real space in the markup, so collapsing
 * runs of whitespace here keeps the spacing uniform however the content is
 * written.
 */
export function runsToWords(runs: readonly TextRun[]): TextWord[] {
  const words: TextWord[] = [];
  for (const run of runs) {
    for (const text of run.text.split(/\s+/)) {
      if (text) {
        words.push(run.className ? { text, className: run.className } : { text });
      }
    }
  }
  return words;
}

/** Seconds between each word. Long headlines reveal faster so the whole
 * sentence still lands quickly instead of crawling word by word. */
export function staggerForWordCount(count: number, base = 0.055): number {
  if (count <= 0) return base;
  const total = base * count;
  return total <= 0.6 ? base : 0.6 / count;
}
