import { motion, useReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

/**
 * Staggered reveal for a list of related items.
 *
 * Ported from motion-primitives' AnimatedGroup (MIT, ibelick/motion-primitives)
 * with three deliberate differences:
 *   - It reveals on scroll (`whileInView`, once) instead of on mount. The
 *     original animates when it mounts, which for a section below the fold
 *     means the animation is over before it is ever seen.
 *   - Reduced motion renders plain elements with no motion components.
 *   - The item is an explicit `AnimatedItem` component rather than the
 *     original's `React.Children.map` + wrapper element. Cloning children
 *     re-keys them by array index and swallows the caller's own element, so
 *     being explicit keeps keys and DOM structure where the caller put them.
 *
 * Framer Motion propagates variants through context, so an `AnimatedItem`
 * does not have to be a direct child of the group.
 */

type GroupTag = "div" | "ul" | "ol";
type ItemTag = "div" | "li";

interface AnimatedGroupProps {
  children: ReactNode;
  as?: GroupTag;
  className?: string;
  /** Seconds between items. */
  stagger?: number;
  /** Seconds before the first item. */
  delay?: number;
  id?: string;
}

const ITEM: Variants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0 },
};

export function AnimatedGroup({
  children,
  as = "div",
  className,
  stagger = 0.06,
  delay = 0,
  id,
}: AnimatedGroupProps) {
  const reduce = useReducedMotion();
  if (reduce) {
    const Tag = as;
    return (
      <Tag className={className} id={id}>
        {children}
      </Tag>
    );
  }

  const MotionTag = as === "ul" ? motion.ul : as === "ol" ? motion.ol : motion.div;
  return (
    <MotionTag
      className={className}
      id={id}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.15 }}
      variants={{
        hidden: {},
        visible: { transition: { delayChildren: delay, staggerChildren: stagger } },
      }}
    >
      {children}
    </MotionTag>
  );
}

interface AnimatedItemProps {
  children: ReactNode;
  as?: ItemTag;
  className?: string;
}

export function AnimatedItem({ children, as = "div", className }: AnimatedItemProps) {
  const reduce = useReducedMotion();
  if (reduce) {
    const Tag = as;
    return <Tag className={className}>{children}</Tag>;
  }

  const MotionTag = as === "li" ? motion.li : motion.div;
  return (
    <MotionTag
      className={className}
      variants={ITEM}
      transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </MotionTag>
  );
}
