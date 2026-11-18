import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useId, useState, type ReactNode } from "react";
import { NARROW_QUERY, defaultGroupOpen } from "../lib/disclosureGate";

/**
 * Collapsible section: a heading that toggles the panel below it.
 *
 * Ported from motion-primitives' Disclosure (MIT, ibelick/motion-primitives),
 * which animates height between 0 and auto inside AnimatePresence. That part
 * is kept; four things are not:
 *   - The original builds its trigger with React.cloneElement and spreads the
 *     caller's props *after* its own, so a child with its own onClick or
 *     className silently loses the toggle. Here the trigger is a real
 *     <button> this component owns.
 *   - It sets an id on the panel but never references it, so the trigger and
 *     panel are not associated. This wires up aria-controls.
 *   - It renders only the first two children (`React.Children.toArray(...)[0]`
 *     and `[1]`), quietly dropping the rest. This takes explicit props.
 *   - Reduced motion skips the height animation and shows or hides outright.
 */

interface DisclosureProps {
  /** Rendered inside the trigger button. */
  summary: ReactNode;
  children: ReactNode;
  /** Heading level wrapping the trigger, so the outline stays correct. */
  headingId?: string;
  headingClassName?: string;
  className?: string;
}

export function Disclosure({
  summary,
  children,
  headingId,
  headingClassName,
  className,
}: DisclosureProps) {
  const reduce = useReducedMotion();
  const panelId = useId();
  // Start open and collapse on narrow screens after mount: the query cannot
  // be read during the first render on the server or in a test environment.
  const [open, setOpen] = useState(true);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const narrow = window.matchMedia(NARROW_QUERY);
    // Only follow the viewport until the reader makes their own choice.
    setOpen(defaultGroupOpen(narrow.matches));
  }, []);

  return (
    <div className={className}>
      <h3 id={headingId} className={headingClassName}>
        <button
          type="button"
          className="disclosure-trigger"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((value) => !value)}
        >
          <span className="disclosure-summary">{summary}</span>
          <svg
            className="disclosure-chevron"
            viewBox="0 0 16 16"
            width="16"
            height="16"
            aria-hidden="true"
            data-open={open}
          >
            <path
              d="M4 6l4 4 4-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </h3>
      {reduce ? (
        <div id={panelId} hidden={!open}>
          {children}
        </div>
      ) : (
        <div className="disclosure-panel">
          <AnimatePresence initial={false}>
            {open && (
              <motion.div
                id={panelId}
                initial="collapsed"
                animate="expanded"
                exit="collapsed"
                variants={{
                  expanded: { height: "auto", opacity: 1 },
                  collapsed: { height: 0, opacity: 0 },
                }}
                transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
              >
                {children}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
