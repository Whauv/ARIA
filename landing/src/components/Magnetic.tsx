import { motion, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { FINE_POINTER_QUERY, REDUCED_MOTION_QUERY, shouldEnableCursor } from "../lib/cursorGate";
import { magneticOffset } from "../lib/magnetic";

/**
 * Nudges its child toward the pointer while the pointer is over it.
 *
 * Ported from motion-primitives' Magnetic (MIT, ibelick/motion-primitives)
 * with four deliberate differences:
 *   - It is gated to fine, hover-capable pointers with reduced motion off,
 *     reusing the same `cursorGate` that decides whether AccentCursor runs.
 *     The original has neither check, so on a touch device the emulated
 *     mouse events still moved the element.
 *   - The listener lives on the element and is attached only while the
 *     pointer is over it. The original keeps a document-level mousemove
 *     listener alive for the lifetime of every instance.
 *   - The pull is weaker and the spring tighter. The original's default
 *     (stiffness 26.7, damping 4.1) overshoots and wobbles, which reads as
 *     ornament rather than feedback.
 *   - The wrapper renders whether or not the effect is enabled, so the
 *     layout is identical in both cases.
 */

const SPRING = { stiffness: 190, damping: 18, mass: 0.12 };

interface MagneticProps {
  children: ReactNode;
  /** Fraction of the pointer offset applied, at most. */
  intensity?: number;
  /** Pull fades to nothing this far from the centre, in pixels. */
  range?: number;
}

export function Magnetic({ children, intensity = 0.32, range = 90 }: MagneticProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [enabled, setEnabled] = useState(false);

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, SPRING);
  const springY = useSpring(y, SPRING);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const fine = window.matchMedia(FINE_POINTER_QUERY);
    const reduced = window.matchMedia(REDUCED_MOTION_QUERY);
    const update = () => {
      const on = shouldEnableCursor({
        finePointer: fine.matches,
        reducedMotion: reduced.matches,
      });
      setEnabled(on);
      if (!on) {
        x.set(0);
        y.set(0);
      }
    };
    update();
    fine.addEventListener("change", update);
    reduced.addEventListener("change", update);
    return () => {
      fine.removeEventListener("change", update);
      reduced.removeEventListener("change", update);
    };
  }, [x, y]);

  const handleMove = (event: React.PointerEvent<HTMLSpanElement>) => {
    if (!enabled || event.pointerType === "touch") return;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const offset = magneticOffset(
      { x: event.clientX, y: event.clientY },
      rect,
      intensity,
      range,
    );
    x.set(offset.x);
    y.set(offset.y);
  };

  const release = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.span
      ref={ref}
      className="magnetic"
      style={{ x: springX, y: springY }}
      onPointerMove={enabled ? handleMove : undefined}
      onPointerLeave={enabled ? release : undefined}
      // A tabbed-to button should not sit off-centre.
      onBlurCapture={release}
    >
      {children}
    </motion.span>
  );
}
