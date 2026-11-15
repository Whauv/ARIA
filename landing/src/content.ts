export const REPO_URL = "https://github.com/Whauv/ARIA";
export const README_URL = "https://github.com/Whauv/ARIA#readme";
export const LICENSE_URL = "https://github.com/Whauv/ARIA/blob/main/LICENSE";

export interface GestureRow {
  gesture: string;
  result: string;
  keys?: string;
}

export interface GestureGroup {
  id: string;
  title: string;
  mode: "draw" | "select" | "any";
  note?: string;
  rows: GestureRow[];
}

export const gestureGroups: GestureGroup[] = [
  {
    id: "g-draw",
    title: "Draw mode",
    mode: "draw",
    rows: [
      { gesture: "Index finger up", result: "Draw", keys: "D enters Draw mode" },
      { gesture: "Index + middle finger up", result: "Lift the pen (move without drawing)" },
      {
        gesture: "Hold a closed fist for 1.5 s",
        result: "Turn the current drawing into a sprite",
      },
      {
        gesture: "Pinch and hold 0.6 s, away from controls",
        result: "Toggle quick color (green ↔ white)",
        keys: "1–8 pick brush colors",
      },
    ],
  },
  {
    id: "g-select",
    title: "Select mode",
    mode: "select",
    rows: [
      {
        gesture: "Pinch a sprite",
        result: "Select it (topmost wins when sprites overlap)",
        keys: "V enters Select mode · click a sprite",
      },
      {
        gesture: "Keep pinching ~0.2 s, then move",
        result: "Drag the sprite; release to place it",
        keys: "Mouse drag",
      },
      { gesture: "Pinch empty space", result: "Deselect", keys: "Esc · click empty space" },
      {
        gesture: "Pinch a layer thumbnail",
        result: "Select that sprite and bring it to front",
        keys: "Click the thumbnail",
      },
      {
        gesture: "Two quick pinches on a sprite, then a third within 1.5 s",
        result: "Delete it (the first two arm delete; undo restores it)",
        keys: "X or Backspace",
      },
      {
        gesture: "Both hands pinching for ~0.3 s",
        result: "Resize the selected sprite around its center",
        keys: "+ / −",
      },
    ],
  },
  {
    id: "g-any",
    title: "Any mode",
    mode: "any",
    note: "Pending holds and drags are cancelled safely — nothing fires.",
    rows: [
      {
        gesture: "Dwell: hover a fingertip over a toolbar button or color swatch and hold",
        result:
          "Activates it — 0.8 s in Draw mode, 0.55 s in Select mode. A bar fills to show progress; it fires once and won't repeat until you move off.",
        keys: "Click toolbar and swatches",
      },
      {
        gesture: "Point at any toolbar button",
        result:
          "Every command has an on-screen button — Draw, Select, Undo, Redo, Clear, Delete, Export, Save, Load, Help, Status — so the keyboard is an alternative, not a requirement. Buttons that cannot act are greyed out.",
        keys: "Each button shows its key",
      },
      {
        gesture: "Point at Close while help is open",
        result:
          "Dismisses the help overlay. Other gestures pause while help is open, but Close still accepts a dwell.",
        keys: "H or Esc",
      },
    ],
  },
];

export interface KeyRow {
  keys: string[];
  action: string;
}

export const keyboard: KeyRow[] = [
  { keys: ["D"], action: "Draw mode" },
  { keys: ["V"], action: "Select mode" },
  { keys: ["1–8"], action: "Brush colors" },
  { keys: ["U"], action: "Undo" },
  { keys: ["R"], action: "Redo" },
  { keys: ["C"], action: "Clear drawing" },
  { keys: ["X", "Backspace"], action: "Delete selected sprite" },
  { keys: ["+", "−"], action: "Resize selected sprite" },
  { keys: ["E"], action: "Export PNG (aria_snapshot.png)" },
  { keys: ["P"], action: "Save project (aria_project.npz)" },
  { keys: ["L"], action: "Load project" },
  { keys: ["H"], action: "Help overlay" },
  { keys: ["S"], action: "Status panel" },
  { keys: ["Esc"], action: "Close help / cancel / deselect" },
  { keys: ["Q"], action: "Quit" },
];

export interface VoiceCommand {
  phrases: string[];
  effect: string;
  cloud?: boolean;
}

export const voiceCommands: VoiceCommand[] = [
  {
    phrases: ["what did I draw", "what is this"],
    effect: "Describes the current frame using Gemini (needs an API key).",
    cloud: true,
  },
  {
    phrases: ["change color to <color>"],
    effect: "red, green, blue, yellow, white, purple, orange, pink, cyan, or eraser.",
  },
  { phrases: ["clear", "clear canvas"], effect: "Clears the drawing." },
  { phrases: ["delete", "remove it"], effect: "Deletes the selected sprite." },
  { phrases: ["make it bigger", "enlarge"], effect: "Enlarges the selected sprite." },
  { phrases: ["make it smaller", "shrink"], effect: "Shrinks the selected sprite." },
  { phrases: ["undo"], effect: "Undoes the last action." },
  { phrases: ["redo"], effect: "Redoes the last undone action." },
  { phrases: ["save"], effect: "Exports a PNG snapshot (aria_snapshot.png)." },
  { phrases: ["draw mode", "select mode"], effect: "Switches mode." },
];

export interface SetupStep {
  label: string;
  code: string;
}

export const setupSteps: SetupStep[] = [
  {
    label: "Clone the repository",
    code: "git clone https://github.com/Whauv/ARIA.git\ncd ARIA",
  },
  { label: "Create a virtual environment (Python 3.12)", code: "python -m venv .venv" },
  { label: "Activate it — Windows (PowerShell)", code: ".\\.venv\\Scripts\\Activate.ps1" },
  { label: "Activate it — macOS / Linux", code: "source .venv/bin/activate" },
  { label: "Install dependencies and run", code: "pip install -r requirements.txt\npython main.py" },
];

export interface EnvVar {
  name: string;
  description: string;
}

export const envVars: EnvVar[] = [
  {
    name: "GOOGLE_API_KEY or GEMINI_API_KEY",
    description: "Enables Gemini scene description (“what did I draw”). Without it, AI stays off.",
  },
  { name: "GEMINI_MODEL", description: "Overrides the Gemini model name." },
  { name: "OPENWAKEWORD_MODEL_PATH", description: "Path to a custom openWakeWord model." },
  {
    name: "OPENWAKEWORD_BUILTIN",
    description: "Built-in wake-word model to use. Default: hey_jarvis.",
  },
  {
    name: "ALLOW_CLOUD_SPEECH_RECOGNITION",
    description:
      "1 (default) allows cloud transcription of voice commands; 0 disables it — the wake word is still detected locally, but spoken commands can't be transcribed.",
  },
];

export const envExample = `# .env (optional)
GOOGLE_API_KEY=your-key-here
# GEMINI_MODEL=
# OPENWAKEWORD_BUILTIN=hey_jarvis
# ALLOW_CLOUD_SPEECH_RECOGNITION=0`;
