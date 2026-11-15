import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// base: "./" keeps asset URLs relative so the build works from any subpath
// (for example GitHub Pages at /ARIA/) or from plain static file hosting.
export default defineConfig({
  base: "./",
  plugins: [react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
