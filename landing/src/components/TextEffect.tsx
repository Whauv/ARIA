import { motion, useReducedMotion, type Variants } from "framer-motion";
import { runsToText, runsToWords, staggerForWordCount, type TextRun } from "../lib/textRuns";

/**
 * Word-by-word reveal for a headline.
 *
 * Ported from motion-primitives' TextEffect (MIT, ibelick/motion-primitives)
 * and adapted in three ways:
 *   - plain CSS instead of Tailwind utility classes;
 *   - reduced motion renders the finished text with no motion components at
 *     all, matching the rest of this site;
 *   - it takes styled runs rather than one string, so parts of the headline
 *     can keep their accent colour.
 *
 * Accessibility departs from the original deliberately. motion-primitives
 * adds a visually-hidden copy of the sentence beside the aria-hidden words,
 * which duplicates the text in the DOM: the heading then reads twice in the
 * accessibility tree and selecting it copies the sentence twice. Naming the
 * element with aria-label instead gives the same single announcement with no
 * duplicate text node.
 */

type Tag = "h1" | "h2" | "p";

interface TextEffectProps {
  runs: readonly TextRun[];
  as?: Tag;
  className?: string;
  id?: string;
  /** Seconds to wait before the first word. */
  delay?: number;
}

const WORD: Variants = {
  hidden: { opacity: 0, y: "0.4em" },
  visible: { opacity: 1, y: 0 },
};

export function TextEffect({ runs, as = "p", className, id, delay = 0 }: TextEffectProps) {
  const reduce = useReducedMotion();
  const text = runsToText(runs);
  const Tag = as;

  if (reduce) {
    return (
      <Tag className={className} id={id}>
        {runs.map((run, index) => (
          <span key={index} className={run.className}>
            {index > 0 ? " " : ""}
            {run.text}
          </span>
        ))}
      </Tag>
    );
  }

  const words = runsToWords(runs);
  const MotionTag = as === "h1" ? motion.h1 : as === "h2" ? motion.h2 : motion.p;

  return (
    <MotionTag
      className={className}
      id={id}
      aria-label={text}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: {
          transition: {
            delayChildren: delay,
            staggerChildren: staggerForWordCount(words.length),
          },
        },
      }}
    >
      {words.map((word, index) => (
        <motion.span
          key={`${index}-${word.text}`}
          aria-hidden="true"
          className={word.className ? `text-effect__word ${word.className}` : "text-effect__word"}
          variants={WORD}
          transition={{ duration: 0.42, ease: [0.2, 0.7, 0.2, 1] }}
        >
          {word.text}
          {index < words.length - 1 ? " " : ""}
        </motion.span>
      ))}
    </MotionTag>
  );
}
