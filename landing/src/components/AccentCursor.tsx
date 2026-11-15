import { motion, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useState } from "react";
import {
  FINE_POINTER_QUERY,
  REDUCED_MOTION_QUERY,
  classifyCursorTarget,
  shouldEnableCursor,
} from "../lib/cursorGate";

/**
 * Small reticle that trails the native cursor on fine-pointer, hover-capable devices.
 * Purely decorative: the native cursor stays visible and this never receives events.
 */
export function AccentCursor() {
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const fine = window.matchMedia(FINE_POINTER_QUERY);
    const reduced = window.matchMedia(REDUCED_MOTION_QUERY);
    const update = () =>
      setEnabled(shouldEnableCursor({ finePointer: fine.matches, reducedMotion: reduced.matches }));
    update();
    fine.addEventListener("change", update);
    reduced.addEventListener("change", update);
    return () => {
      fine.removeEventListener("change", update);
      reduced.removeEventListener("change", update);
    };
  }, []);

  return enabled ? <Reticle /> : null;
}

function Reticle() {
  const x = useMotionValue(-100);
  const y = useMotionValue(-100);
  const scaleTarget = useMotionValue(1);
  const opacityTarget = useMotionValue(0);

  const sx = useSpring(x, { stiffness: 600, damping: 45, mass: 0.35 });
  const sy = useSpring(y, { stiffness: 600, damping: 45, mass: 0.35 });
  const scale = useSpring(scaleTarget, { stiffness: 400, damping: 30 });
  const opacity = useSpring(opacityTarget, { stiffness: 400, damping: 40 });

  useEffect(() => {
    let seen = false;
    let overEditable = false;

    const onMove = (e: PointerEvent) => {
      if (e.pointerType && e.pointerType !== "mouse" && e.pointerType !== "pen") {
        opacityTarget.set(0);
        return;
      }
      x.set(e.clientX);
      y.set(e.clientY);
      if (!seen) {
        // First sighting: place without trailing in from the corner.
        sx.jump(e.clientX);
        sy.jump(e.clientY);
        seen = true;
      }
      opacityTarget.set(overEditable ? 0 : 0.85);
    };

    // Delegated: one listener classifies whatever the pointer enters.
    const onOver = (e: PointerEvent) => {
      const kind = classifyCursorTarget(e.target);
      overEditable = kind === "hidden";
      scaleTarget.set(kind === "interactive" ? 1.5 : 1);
      opacityTarget.set(overEditable ? 0 : 0.85);
    };

    const hide = () => opacityTarget.set(0);
    const onOut = (e: MouseEvent) => {
      if (!e.relatedTarget) hide();
    };
    const onVisibility = () => {
      if (document.hidden) hide();
    };

    document.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("pointerover", onOver, { passive: true });
    document.documentElement.addEventListener("pointerleave", hide);
    document.addEventListener("mouseout", onOut);
    window.addEventListener("blur", hide);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerover", onOver);
      document.documentElement.removeEventListener("pointerleave", hide);
      document.removeEventListener("mouseout", onOut);
      window.removeEventListener("blur", hide);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [x, y, sx, sy, scaleTarget, opacityTarget]);

  return (
    <motion.div className="accent-cursor" aria-hidden="true" style={{ x: sx, y: sy, scale, opacity }}>
      <svg viewBox="0 0 28 28">
        <circle cx="14" cy="14" r="8" fill="none" stroke="#5EC8F2" strokeWidth="1.25" />
        <path d="M14 1v4M14 23v4M1 14h4M23 14h4" stroke="#A3A7AE" strokeWidth="1" />
      </svg>
    </motion.div>
  );
}
