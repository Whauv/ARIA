import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

const C = {
  bg: "#0E1012",
  g800: "#1E2228",
  g700: "#2A2F37",
  g500: "#5A616C",
  text: "#E8E6E1",
  muted: "#A3A7AE",
  draw: "#F5B841",
  select: "#5EC8F2",
  ok: "#6FD39B",
  warn: "#F08A4B",
};

const ease = [0.45, 0, 0.2, 1] as const;

/** Props for a path that draws itself once when scrolled into view. */
function useDraw() {
  const reduce = useReducedMotion();
  return (delay = 0, duration = 1.1) => ({
    initial: reduce ? false : { pathLength: 0 },
    whileInView: { pathLength: 1 },
    viewport: { once: true, amount: 0.6 },
    transition: { duration, delay, ease },
  });
}

/** Props for an element that fades in once when in view. */
function useAppear() {
  const reduce = useReducedMotion();
  return (delay = 0) => ({
    initial: reduce ? false : { opacity: 0 },
    whileInView: { opacity: 1 },
    viewport: { once: true, amount: 0.6 },
    transition: { duration: 0.35, delay },
  });
}

function Frame({ label, coord, children }: { label: string; coord: string; children: ReactNode }) {
  const pid = `grid-${coord.replace(/[^a-z0-9]+/gi, "-")}`;
  return (
    <div className="ticked">
      <svg viewBox="0 0 320 200" role="img" aria-label={label}>
        <defs>
          <pattern id={pid} width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="10" cy="10" r="0.8" fill={C.g700} />
          </pattern>
        </defs>
        <rect x="0" y="0" width="320" height="200" fill={`url(#${pid})`} />
        <text x="10" y="16" fontSize="8" letterSpacing="1" fill={C.muted}>
          {coord}
        </text>
        {children}
      </svg>
    </div>
  );
}

function Label({ x, y, children, color = C.muted, anchor = "start" }: {
  x: number;
  y: number;
  children: ReactNode;
  color?: string;
  anchor?: "start" | "middle" | "end";
}) {
  return (
    <text x={x} y={y} fontSize="8.5" letterSpacing="0.9" fill={color} textAnchor={anchor}>
      {children}
    </text>
  );
}

/** Schematic hand: palm plus extended fingers (index, middle) as rounded strokes. */
function Hand({ x, y, fingers, color }: { x: number; y: number; fingers: ("index" | "middle")[]; color: string }) {
  return (
    <g transform={`translate(${x} ${y})`} stroke={color} strokeWidth="1.5" fill="none" strokeLinecap="round">
      <rect x="-13" y="0" width="26" height="24" rx="7" fill={C.bg} />
      {/* folded finger knuckles */}
      <path d="M-9 0 v-4 M-3 0 v-4 M3 0 v-4 M9 0 v-4" stroke={C.g500} />
      {fingers.includes("index") && <line x1="-9" y1="-4" x2="-9" y2="-30" strokeWidth="5" stroke={color} />}
      {fingers.includes("middle") && <line x1="-3" y1="-4" x2="-3" y2="-32" strokeWidth="5" stroke={color} />}
      <path d="M13 12 q8 -2 8 -10" />
    </g>
  );
}

function Fist({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <g transform={`translate(${x} ${y})`} stroke={color} strokeWidth="1.5" fill={C.bg} strokeLinecap="round">
      <rect x="-14" y="-12" width="28" height="26" rx="8" />
      <path d="M-14 -3 h28 M-5 -12 v9 M4 -12 v9" fill="none" stroke={C.g500} />
    </g>
  );
}

function Handles({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <>
      {[
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h],
      ].map(([hx, hy], i) => (
        <rect key={i} x={hx - 3} y={hy - 3} width="6" height="6" fill={C.bg} stroke={C.select} strokeWidth="1.5" />
      ))}
    </>
  );
}

const SQUIGGLE = "M28 112 C36 72 76 52 108 66 C140 80 132 116 106 118 C86 120 88 92 114 86";

export function DrawDiagram() {
  const draw = useDraw();
  const appear = useAppear();
  return (
    <Frame label="Index finger up draws a stroke; index and middle finger up lifts the pen." coord="01 · DRAW">
      <motion.path
        d={SQUIGGLE}
        fill="none"
        stroke={C.draw}
        strokeWidth="3.5"
        strokeLinecap="round"
        {...draw(0.1, 1.2)}
      />
      <circle cx="114" cy="86" r="6" fill="none" stroke={C.draw} strokeWidth="1.25" />
      <Hand x={123} y={120} fingers={["index"]} color={C.draw} />
      <motion.g {...appear(1.1)}>
        <line x1="119" y1="81" x2="132" y2="56" stroke={C.g500} />
        <line x1="132" y1="56" x2="146" y2="56" stroke={C.g500} />
        <Label x={100} y={44}>
          FINGERTIP
        </Label>
      </motion.g>
      <Label x={20} y={180} color={C.text}>
        INDEX UP → DRAW
      </Label>

      <line x1="168.5" y1="34" x2="168.5" y2="186" stroke={C.g700} strokeDasharray="2 4" />

      {/* pen lifted: the fingertip moves but leaves only a faint path */}
      <path
        d="M186 84 C210 70 240 70 268 92"
        fill="none"
        stroke={C.muted}
        strokeWidth="1.25"
        strokeDasharray="1 5"
        strokeLinecap="round"
      />
      <circle cx="268" cy="92" r="6" fill="none" stroke={C.text} strokeWidth="1.25" />
      <Hand x={274} y={126} fingers={["index", "middle"]} color={C.text} />
      <Label x={184} y={180} color={C.text}>
        2 FINGERS UP
      </Label>
      <Label x={184} y={192}>
        → PEN LIFTED
      </Label>
    </Frame>
  );
}

export function SpriteDiagram() {
  const draw = useDraw();
  const appear = useAppear();
  return (
    <Frame label="Holding a closed fist for 1.5 seconds turns the drawing into a sprite." coord="02 · SPRITE">
      <path d="M40 120 C50 80 90 64 120 78 C150 92 140 126 116 128" fill="none" stroke={C.draw} strokeWidth="3" strokeLinecap="round" />
      <Fist x={92} y={160} color={C.draw} />
      <circle cx="92" cy="160" r="24" fill="none" stroke={C.g700} strokeWidth="2" />
      <motion.circle
        cx="92"
        cy="160"
        r="24"
        fill="none"
        stroke={C.draw}
        strokeWidth="2"
        strokeLinecap="round"
        transform="rotate(-90 92 160)"
        {...draw(0.2, 1.5)}
      />
      <Label x={124} y={176} color={C.text}>
        HOLD 1.5 s
      </Label>

      <motion.g {...appear(1.7)}>
        <line x1="160" y1="100" x2="186" y2="100" stroke={C.g500} />
        <path d="M182 96 l5 4 -5 4" fill="none" stroke={C.g500} />
        <path d="M206 112 C214 82 244 70 266 80 C288 90 280 116 262 118" fill="none" stroke={C.draw} strokeWidth="3" strokeLinecap="round" />
        <rect x="198" y="64" width="98" height="64" fill="none" stroke={C.muted} strokeDasharray="5 4" />
        <Label x={198} y={56} color={C.text}>
          SPRITE
        </Label>
        <Label x={198} y={146}>
          MOVE · RESIZE · DELETE
        </Label>
      </motion.g>
    </Frame>
  );
}

export function SelectDiagram() {
  const draw = useDraw();
  const appear = useAppear();
  const reduce = useReducedMotion();
  return (
    <Frame label="Pinch a sprite to select it, keep pinching to drag, release to place." coord="03 · SELECT">
      {/* origin ghost */}
      <rect x="34" y="62" width="96" height="64" fill="none" stroke={C.g500} strokeDasharray="3 4" />
      <Label x={34} y={142}>
        PICKED UP
      </Label>

      <motion.g
        initial={reduce ? false : { x: -150 }}
        whileInView={{ x: 0 }}
        viewport={{ once: true, amount: 0.6 }}
        transition={{ duration: 1, delay: 0.6, ease }}
      >
        <path d="M194 112 C202 82 232 70 254 80 C276 90 268 116 250 118" fill="none" stroke={C.draw} strokeWidth="3" strokeLinecap="round" />
        <motion.rect x="184" y="62" width="96" height="64" fill="none" stroke={C.select} strokeWidth="1.5" {...draw(0, 0.5)} />
        <motion.g {...appear(0.4)}>
          <Handles x={184} y={62} w={96} h={64} />
        </motion.g>
        {/* pinch: thumb + index tips touching */}
        <circle cx="228" cy="96" r="5" fill="none" stroke={C.select} strokeWidth="1.5" />
        <circle cx="236" cy="100" r="5" fill="none" stroke={C.text} strokeWidth="1.5" />
      </motion.g>

      <motion.g {...appear(1.6)}>
        <line x1="240" y1="128" x2="252" y2="160" stroke={C.g500} />
        <Label x={160} y={174} color={C.text}>
          PINCH ~0.2 s → DRAG
        </Label>
        <Label x={160} y={188}>
          RELEASE TO PLACE
        </Label>
      </motion.g>
    </Frame>
  );
}

export function VoiceDiagram() {
  const appear = useAppear();
  const bars = [6, 14, 22, 12, 26, 18, 8, 16, 10];
  return (
    <Frame label="Say the wake phrase, then a command. The wake word is detected locally; commands are transcribed in the cloud." coord="04 · VOICE">
      <g transform="translate(24 60)">
        {bars.map((h, i) => (
          <rect key={i} x={i * 7} y={-h / 2} width="3" height={h} rx="1.5" fill={C.muted} />
        ))}
      </g>
      <Label x={24} y={96} color={C.text}>
        “HEY JARVIS”
      </Label>
      <g>
        <circle cx="28" cy="110" r="3" fill={C.ok} />
        <Label x={36} y={113}>
          WAKE WORD · LOCAL
        </Label>
      </g>

      <motion.g {...appear(0.4)}>
        <line x1="96" y1="60" x2="128" y2="60" stroke={C.g500} />
        <path d="M124 56 l5 4 -5 4" fill="none" stroke={C.g500} />
        <Label x={140} y={58} color={C.text}>
          “WHAT DID I DRAW”
        </Label>
        <circle cx="144" cy="74" r="3" fill={C.warn} />
        <Label x={152} y={77}>
          TRANSCRIBED · CLOUD
        </Label>
      </motion.g>

      <motion.g {...appear(0.9)}>
        <line x1="190" y1="88" x2="190" y2="118" stroke={C.g500} />
        <path d="M186 114 l4 5 4 -5" fill="none" stroke={C.g500} />
        <rect x="140" y="126" width="116" height="44" fill="none" stroke={C.g700} />
        <rect x="146" y="132" width="44" height="32" fill={C.g800} />
        <path d="M152 156 C156 140 172 138 178 146" fill="none" stroke={C.draw} strokeWidth="2" strokeLinecap="round" />
        <Label x={196} y={146} color={C.text}>
          1 FRAME
        </Label>
        <Label x={196} y={158}>
          → GEMINI
        </Label>
        <circle cx="144" cy="186" r="3" fill={C.warn} />
        <Label x={152} y={189}>
          ONLY IF AN API KEY IS SET
        </Label>
      </motion.g>
    </Frame>
  );
}
