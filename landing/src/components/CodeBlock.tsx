import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { browserCopyDeps, copyText } from "../lib/clipboard";
import { useAnnounce } from "./Announcer";

type State = "idle" | "copied" | "failed";

interface CodeBlockProps {
  code: string;
  /** Accessible name for what is being copied, e.g. "install commands". */
  label: string;
}

export function CodeBlock({ code, label }: CodeBlockProps) {
  const [state, setState] = useState<State>("idle");
  const announce = useAnnounce();
  const reduce = useReducedMotion();
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function onCopy() {
    const result = await copyText(code, browserCopyDeps());
    const next: State = result === "failed" ? "failed" : "copied";
    setState(next);
    announce(next === "copied" ? "Copied" : "Copy failed — select the text and copy it manually");
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState("idle"), 1800);
  }

  const text = state === "copied" ? "Copied" : state === "failed" ? "Failed" : "Copy";

  return (
    <div className="codeblock">
      <pre tabIndex={0} aria-label={label}>
        <code>{code}</code>
      </pre>
      <button
        type="button"
        className="copy-btn"
        data-state={state}
        onClick={onCopy}
        aria-label={`Copy ${label}`}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          {state === "copied" ? (
            <motion.path
              d="M2 6.5 L5 9 L10 3"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={reduce ? false : { pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.3 }}
            />
          ) : (
            <path
              d="M4 1.5 h6.5 v6.5 M1.5 4 h6.5 v6.5 h-6.5 z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
            />
          )}
        </svg>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={text}
            initial={reduce ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
          >
            {text}
          </motion.span>
        </AnimatePresence>
      </button>
    </div>
  );
}
