# ARIA website

Static documentation site for ARIA, built with React, TypeScript, Vite, and framer-motion.
It describes the desktop app and does not run it: no camera or microphone access.

Frontend dependencies live only in this folder. The Python app does not depend on them.

## Commands

Run these from `landing/`:

```sh
npm install        # install dependencies
npm run dev        # start the dev server with hot reload
npm run typecheck  # tsc --noEmit
npm test           # vitest run (unit tests for src/lib)
npm run build      # typecheck, then build to dist/
npm run preview    # serve the built dist/ locally
```

## Deploying

`vite.config.ts` sets `base: "./"`, so asset URLs in `dist/` are relative. You can host the
folder at any path (for example GitHub Pages at `/ARIA/`) on any static file host. Browsers
block module scripts on `file://`, so open the build through a server such as `npm run preview`.

## Layout

- `src/content.ts`: product facts (gestures, keys, voice commands, setup, `.env`). Keep this in sync with the app.
- `src/components/`: page sections, SVG illustrations, and the accent cursor
- `src/lib/`: pure helpers (clipboard copy with fallback, cursor capability gate) and their tests
- `src/styles.css`: design tokens (shared with the Python app) and layout

## Motion and accessibility

- `MotionConfig reducedMotion="user"` and `useReducedMotion` make every animation respect
  `prefers-reduced-motion`. With reduced motion on, illustrations render in their final state.
- The accent cursor only runs on hover-capable fine pointers, and only when reduced motion is off.
  It does not replace the native cursor, and it ignores pointer events.
