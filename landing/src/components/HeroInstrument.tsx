import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
  type AnimationPlaybackControls,
  type MotionValue,
} from "framer-motion";
import { useEffect, useRef, useState } from "react";

const C = {
  bg: "#0E1012",
  panel: "#15181C",
  g800: "#1E2228",
  g700: "#2A2F37",
  g500: "#5A616C",
  text: "#E8E6E1",
  muted: "#A3A7AE",
  draw: "#F5B841",
  select: "#5EC8F2",
  ok: "#6FD39B",
};

/** Phases of the illustrated sequence. */
const P = { Draw: 0, Fist: 1, Sprite: 2, Dwell: 3, Select: 4, Drag: 5, Done: 6 } as const;
type P = (typeof P)[keyof typeof P];

const STROKE =
  "M214 300 C196 240 236 176 300 172 C364 168 404 214 388 262 C374 304 318 318 292 286 C268 256 300 222 336 232 C372 242 410 222 446 188";
const BOX = { x: 192, y: 158, w: 266, h: 166 };
const STROKE_START = { x: 214, y: 300 };
const STROKE_END = { x: 446, y: 188 };
const SELECT_BTN = { x: 34, y: 83 };
const GRAB = { x: 330, y: 250 };
const DRAG = { x: -40, y: 26 };

// Mirrors the real app toolbar (src/aria/config.py TOOLBAR_ACTIONS).
const TOOLS = [
  "DRAW",
  "SELECT",
  "UNDO",
  "REDO",
  "CLEAR",
  "DELETE",
  "EXPORT",
  "SAVE",
  "LOAD",
  "HELP",
  "STATUS",
];
const SWATCHES = ["#6FD39B", "#E8E6E1", "#F0605D", "#4F86F7", "#F5B841", "#A77BF3", "#F08A4B", "#F38BC4"];
const SERVICES: { label: string; on: boolean; x: number; w: number }[] = [
  { label: "HAND", on: true, x: 444, w: 52 },
  { label: "VOICE", on: true, x: 502, w: 56 },
  { label: "AI OFF", on: false, x: 562, w: 62 },
];

const ease = [0.45, 0, 0.2, 1] as const;

export function HeroInstrument() {
  const reduce = useReducedMotion();
  const [phase, setPhase] = useState<P>(reduce ? P.Done : P.Draw);
  const [run, setRun] = useState(0);
  const pathRef = useRef<SVGPathElement>(null);

  const draw = useMotionValue(reduce ? 1 : 0);
  const fist = useMotionValue(reduce ? 1 : 0);
  const dwell = useMotionValue(reduce ? 1 : 0);
  const fx = useMotionValue(STROKE_START.x);
  const fy = useMotionValue(STROKE_START.y);
  const dx = useMotionValue(0);
  const dy = useMotionValue(0);

  useEffect(() => {
    const finalState = () => {
      draw.set(1);
      fist.set(1);
      dwell.set(1);
      dx.set(DRAG.x);
      dy.set(DRAG.y);
      fx.set(GRAB.x + DRAG.x);
      fy.set(GRAB.y + DRAG.y);
      setPhase(P.Done);
    };

    if (reduce) {
      finalState();
      return;
    }

    let cancelled = false;
    const controls: AnimationPlaybackControls[] = [];
    const timers: number[] = [];
    const go = (mv: MotionValue<number>, to: number, duration: number, delay = 0) => {
      const c = animate(mv, to, { duration, delay, ease });
      controls.push(c);
      return c;
    };
    const wait = (ms: number) =>
      new Promise<void>((resolve) => {
        timers.push(window.setTimeout(resolve, ms));
      });

    // Reset for (re)play.
    draw.set(0);
    fist.set(0);
    dwell.set(0);
    dx.set(0);
    dy.set(0);
    fx.set(STROKE_START.x);
    fy.set(STROKE_START.y);
    setPhase(P.Draw);

    const path = pathRef.current;
    const length = path?.getTotalLength() ?? 0;
    const unfollow = draw.on("change", (p) => {
      if (!path) return;
      const pt = path.getPointAtLength(p * length);
      fx.set(pt.x);
      fy.set(pt.y);
    });

    (async () => {
      await wait(500);
      if (cancelled) return;
      await go(draw, 1, 2.2);
      unfollow();
      if (cancelled) return;
      setPhase(P.Fist);
      await go(fist, 1, 1.5, 0.15); // closed fist held 1.5 s
      if (cancelled) return;
      setPhase(P.Sprite);
      await wait(700);
      if (cancelled) return;
      setPhase(P.Dwell);
      await Promise.all([go(fx, SELECT_BTN.x, 0.7), go(fy, SELECT_BTN.y, 0.7)]);
      if (cancelled) return;
      await go(dwell, 1, 0.8); // dwell 0.8 s in Draw mode
      if (cancelled) return;
      setPhase(P.Select);
      await Promise.all([go(fx, GRAB.x, 0.7, 0.2), go(fy, GRAB.y, 0.7, 0.2)]);
      if (cancelled) return;
      setPhase(P.Drag);
      await wait(350);
      if (cancelled) return;
      await Promise.all([
        go(dx, DRAG.x, 0.9),
        go(dy, DRAG.y, 0.9),
        go(fx, GRAB.x + DRAG.x, 0.9),
        go(fy, GRAB.y + DRAG.y, 0.9),
      ]);
      if (cancelled) return;
      setPhase(P.Done);
    })();

    return () => {
      cancelled = true;
      unfollow();
      controls.forEach((c) => c.stop());
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [reduce, run, draw, fist, dwell, fx, fy, dx, dy]);

  const mode = phase >= P.Select ? "select" : "draw";
  const modeColor = mode === "select" ? C.select : C.draw;
  const isSprite = phase >= P.Sprite;
  const isSelected = phase >= P.Select;
  const fingerLabel =
    phase === P.Draw
      ? "INDEX UP · DRAW"
      : phase === P.Fist
        ? "FIST · HOLD 1.5 s"
        : phase === P.Sprite
          ? ""
          : phase === P.Dwell
            ? "DWELL 0.8 s"
            : phase === P.Select
              ? "PINCH · SELECT"
              : phase === P.Drag
                ? "PINCH · DRAG"
                : "";

  const labelX = useTransform(fx, (v) => v + 14);
  const labelY = useTransform(fy, (v) => v - 14);

  return (
    <figure className="figure">
      <div className="ticked">
        <svg
          className="instrument"
          viewBox="0 0 640 420"
          role="img"
          aria-labelledby="instrument-title instrument-desc"
        >
          <title id="instrument-title">Illustration of the ARIA app window</title>
          <desc id="instrument-desc">
            A stroke is drawn with an index fingertip in Draw mode. Holding a closed fist turns it
            into a sprite. Dwelling on the Select toolbar button switches to Select mode, where a
            pinch selects the sprite — shown with a cyan outline and corner handles — and drags it.
          </desc>

          <defs>
            <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse" x="68" y="48">
              <circle cx="12" cy="12" r="0.9" fill={C.g700} />
            </pattern>
            <path id="stroke-shape" d={STROKE} />
          </defs>

          {/* Window */}
          <rect x="0.5" y="0.5" width="639" height="419" fill={C.panel} stroke={C.g700} />
          <line x1="0" y1="40.5" x2="640" y2="40.5" stroke={C.g700} />
          <text x="18" y="25" fill={C.text} fontSize="12" fontWeight="700" letterSpacing="2.5">
            ARIA
          </text>

          {/* Mode chip */}
          <rect x="66" y="12" width="78" height="18" rx="9" fill="none" stroke={C.g700} />
          <motion.circle cx="78" cy="21" r="3" initial={false} animate={{ fill: modeColor }} />
          <motion.text
            x="86"
            y="24.5"
            fontSize="9"
            letterSpacing="1.4"
            initial={false}
            animate={{ fill: modeColor }}
          >
            {mode === "select" ? "SELECT" : "DRAW"}
          </motion.text>

          {/* Service chips */}
          {SERVICES.map((s) => (
            <g key={s.label}>
              <rect x={s.x} y="12" width={s.w} height="18" rx="9" fill="none" stroke={C.g700} />
              <circle cx={s.x + 11} cy="21" r="2.6" fill={s.on ? C.ok : C.g500} />
              <text x={s.x + 19} y="24.5" fontSize="8.5" letterSpacing="1" fill={C.muted}>
                {s.label}
              </text>
            </g>
          ))}

          {/* Toolbar */}
          {TOOLS.map((t, i) => {
            const y = 52 + i * 22;
            const active = (t === "DRAW" && mode === "draw") || (t === "SELECT" && mode === "select");
            return (
              <g key={t}>
                <motion.rect
                  x="12.5"
                  y={y + 0.5}
                  width="44"
                  height="18"
                  rx="3"
                  fill={C.g800}
                  initial={false}
                  animate={{ stroke: active ? modeColor : C.g700 }}
                />
                <text
                  x="34.5"
                  y={y + 12.5}
                  fontSize="6.5"
                  letterSpacing="0.8"
                  textAnchor="middle"
                  fill={active ? C.text : C.muted}
                >
                  {t}
                </text>
              </g>
            );
          })}

          {/* Dwell ring on the Select button */}
          {phase === P.Dwell && (
            <g>
              <circle cx={SELECT_BTN.x} cy={SELECT_BTN.y} r="19" fill="none" stroke={C.g700} strokeWidth="2" />
              <motion.circle
                cx={SELECT_BTN.x}
                cy={SELECT_BTN.y}
                r="19"
                fill="none"
                stroke={C.select}
                strokeWidth="2"
                strokeLinecap="round"
                transform={`rotate(-90 ${SELECT_BTN.x} ${SELECT_BTN.y})`}
                style={{ pathLength: dwell }}
              />
            </g>
          )}

          {/* Swatches */}
          {SWATCHES.map((c, i) => {
            const cx = 24 + (i % 2) * 21;
            const cy = 310 + Math.floor(i / 2) * 22;
            return (
              <g key={c}>
                <circle cx={cx} cy={cy} r="6.5" fill={c} />
                {c === C.draw && (
                  <circle cx={cx} cy={cy} r="9.5" fill="none" stroke={C.text} strokeWidth="1" />
                )}
              </g>
            );
          })}

          {/* Canvas */}
          <rect x="68.5" y="48.5" width="559" height="359" fill={C.bg} stroke={C.g700} />
          <rect x="69" y="49" width="558" height="358" fill="url(#dots)" />
          {Array.from({ length: 11 }, (_, i) => (
            <line
              key={`t${i}`}
              x1={68 + (i + 1) * 48}
              y1="49"
              x2={68 + (i + 1) * 48}
              y2="54"
              stroke={C.g500}
            />
          ))}
          {Array.from({ length: 7 }, (_, i) => (
            <line
              key={`l${i}`}
              x1="69"
              y1={48 + (i + 1) * 48}
              x2="74"
              y2={48 + (i + 1) * 48}
              stroke={C.g500}
            />
          ))}

          {/* Layer thumbnails */}
          <text x="560" y="346" fontSize="7" letterSpacing="1" fill={C.muted}>
            LAYERS
          </text>
          <rect x="560.5" y="352.5" width="58" height="42" fill="none" stroke={C.g700} strokeDasharray="2 3" />
          {isSprite && (
            <motion.g
              initial={reduce ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4 }}
            >
              <rect
                x="560.5"
                y="352.5"
                width="58"
                height="42"
                fill={C.panel}
                stroke={isSelected ? C.select : C.g500}
              />
              <use
                href="#stroke-shape"
                transform={`translate(563 357) scale(0.2) translate(${-BOX.x} ${-BOX.y})`}
                fill="none"
                stroke={C.draw}
                strokeWidth="18"
                strokeLinecap="round"
              />
            </motion.g>
          )}

          {/* Where the sprite was picked up from */}
          {(phase === P.Drag || phase === P.Done) && (
            <rect
              x={BOX.x}
              y={BOX.y}
              width={BOX.w}
              height={BOX.h}
              fill="none"
              stroke={C.g500}
              strokeDasharray="3 4"
              opacity="0.6"
            />
          )}

          {/* Sprite (stroke + outline) moves together when dragged */}
          <motion.g style={{ x: dx, y: dy }}>
            <motion.path
              ref={pathRef}
              d={STROKE}
              fill="none"
              stroke={C.draw}
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ pathLength: draw }}
            />

            {/* Fist progress → becomes a sprite (unselected bounds, dashed) */}
            {isSprite && !isSelected && (
              <motion.rect
                x={BOX.x}
                y={BOX.y}
                width={BOX.w}
                height={BOX.h}
                fill="none"
                stroke={C.muted}
                strokeWidth="1"
                strokeDasharray="5 4"
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.35 }}
              />
            )}

            {/* Selection: 1.5px cyan outline + square corner handles */}
            {isSelected && (
              <g>
                <motion.rect
                  x={BOX.x}
                  y={BOX.y}
                  width={BOX.w}
                  height={BOX.h}
                  fill="none"
                  stroke={C.select}
                  strokeWidth="1.5"
                  initial={reduce ? false : { pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.45, ease }}
                />
                {[
                  [BOX.x, BOX.y],
                  [BOX.x + BOX.w, BOX.y],
                  [BOX.x, BOX.y + BOX.h],
                  [BOX.x + BOX.w, BOX.y + BOX.h],
                ].map(([hx, hy], i) => (
                  <motion.rect
                    key={i}
                    x={hx - 3.5}
                    y={hy - 3.5}
                    width="7"
                    height="7"
                    fill={C.bg}
                    stroke={C.select}
                    strokeWidth="1.5"
                    initial={reduce ? false : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 + i * 0.05, duration: 0.2 }}
                  />
                ))}
              </g>
            )}

            {/* Annotation leader line */}
            {isSprite && (
              <motion.g
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: 0.1 }}
              >
                <polyline
                  points={`${BOX.x + BOX.w + 4},${BOX.y - 4} ${BOX.x + BOX.w + 22},${BOX.y - 22} ${BOX.x + BOX.w + 70},${BOX.y - 22}`}
                  fill="none"
                  stroke={isSelected ? C.select : C.g500}
                  strokeWidth="1"
                />
                <text
                  x={BOX.x + BOX.w + 26}
                  y={BOX.y - 27}
                  fontSize="8.5"
                  letterSpacing="1"
                  fill={isSelected ? C.select : C.muted}
                >
                  {isSelected ? "SPRITE · SELECTED" : "SPRITE"}
                </text>
                <text x={BOX.x + BOX.w + 26} y={BOX.y - 10} fontSize="7.5" letterSpacing="0.6" fill={C.muted}>
                  {`w ${BOX.w} · h ${BOX.h}`}
                </text>
              </motion.g>
            )}
          </motion.g>

          {/* Fist hold progress ring around the fingertip */}
          {phase === P.Fist && (
            <g>
              <circle cx={STROKE_END.x} cy={STROKE_END.y} r="15" fill="none" stroke={C.g700} strokeWidth="2" />
              <motion.circle
                cx={STROKE_END.x}
                cy={STROKE_END.y}
                r="15"
                fill="none"
                stroke={C.draw}
                strokeWidth="2"
                strokeLinecap="round"
                transform={`rotate(-90 ${STROKE_END.x} ${STROKE_END.y})`}
                style={{ pathLength: fist }}
              />
            </g>
          )}

          {/* Fingertip reticle */}
          <motion.g style={{ x: fx, y: fy }}>
            <motion.circle
              r={isSelected ? 6 : 8}
              fill="none"
              initial={false}
              animate={{ stroke: modeColor }}
              strokeWidth="1.5"
            />
            <circle r="1.8" fill={C.text} />
            <line x1="-13" y1="0" x2="-10" y2="0" stroke={C.muted} />
            <line x1="10" y1="0" x2="13" y2="0" stroke={C.muted} />
            <line x1="0" y1="-13" x2="0" y2="-10" stroke={C.muted} />
            <line x1="0" y1="10" x2="0" y2="13" stroke={C.muted} />
          </motion.g>
          {fingerLabel && (
            <motion.text
              x={labelX}
              y={labelY}
              fontSize="8.5"
              letterSpacing="1"
              fill={C.text}
              stroke={C.bg}
              strokeWidth="3"
              paintOrder="stroke"
            >
              {fingerLabel}
            </motion.text>
          )}
        </svg>
      </div>
      <figcaption>
        <span>Illustration — not a live demo. Nothing here uses your camera or microphone.</span>
        {!reduce && (
          <button
            type="button"
            className="btn btn--ghost"
            style={{ padding: "0.3rem 0.7rem", fontSize: "0.75rem", fontFamily: "var(--font-mono)" }}
            onClick={() => setRun((r) => r + 1)}
          >
            Replay
          </button>
        )}
      </figcaption>
    </figure>
  );
}
