import { describe, expect, it, vi } from "vitest";
import { copyText, legacyCopy, type LegacyCopyDocument } from "./clipboard";

describe("copyText", () => {
  it("uses the Clipboard API when it succeeds and skips the fallback", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const fallback = vi.fn().mockReturnValue(true);
    await expect(copyText("pip install", { writeText, legacyCopy: fallback })).resolves.toBe(
      "clipboard",
    );
    expect(writeText).toHaveBeenCalledWith("pip install");
    expect(fallback).not.toHaveBeenCalled();
  });

  it("falls back to legacy copy when the Clipboard API rejects", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("NotAllowedError"));
    const fallback = vi.fn().mockReturnValue(true);
    await expect(copyText("python main.py", { writeText, legacyCopy: fallback })).resolves.toBe(
      "fallback",
    );
    expect(fallback).toHaveBeenCalledWith("python main.py");
  });

  it("uses legacy copy when no Clipboard API is available", async () => {
    await expect(copyText("x", { legacyCopy: () => true })).resolves.toBe("fallback");
  });

  it("reports failure instead of throwing when every path fails", async () => {
    const result = await copyText("x", {
      writeText: () => Promise.reject(new Error("nope")),
      legacyCopy: () => {
        throw new Error("execCommand blew up");
      },
    });
    expect(result).toBe("failed");
    await expect(copyText("x", { legacyCopy: () => false })).resolves.toBe("failed");
    await expect(copyText("x", {})).resolves.toBe("failed");
  });
});

function fakeDocument(execResult: boolean | Error) {
  const attached: unknown[] = [];
  const focus = vi.fn();
  const area = {
    value: "",
    style: {} as Record<string, string>,
    setAttribute: vi.fn(),
    select: vi.fn(),
  };
  const raw = {
    activeElement: { focus },
    createElement: vi.fn(() => area),
    body: {
      appendChild: vi.fn((n: unknown) => attached.push(n)),
      removeChild: vi.fn((n: unknown) => attached.splice(attached.indexOf(n), 1)),
    },
    execCommand: vi.fn((): boolean => {
      if (execResult instanceof Error) throw execResult;
      return execResult;
    }),
  };
  return { doc: raw as unknown as LegacyCopyDocument, raw, area, attached, focus };
}

describe("legacyCopy", () => {
  it("copies from an off-screen textarea, cleans up, and restores focus", () => {
    const { doc, raw, area, attached, focus } = fakeDocument(true);
    expect(legacyCopy("git clone", doc)).toBe(true);
    expect(area.value).toBe("git clone");
    expect(area.style.position).toBe("fixed");
    expect(area.select).toHaveBeenCalled();
    expect(raw.execCommand).toHaveBeenCalledWith("copy");
    expect(attached).toHaveLength(0);
    expect(focus).toHaveBeenCalled();
  });

  it("still removes the textarea when execCommand throws", () => {
    const { doc, attached } = fakeDocument(new Error("blocked"));
    expect(legacyCopy("x", doc)).toBe(false);
    expect(attached).toHaveLength(0);
  });

  it("returns false when execCommand reports failure", () => {
    const { doc } = fakeDocument(false);
    expect(legacyCopy("x", doc)).toBe(false);
  });

  it("returns false without touching the DOM when execCommand is unavailable", () => {
    const { raw } = fakeDocument(true);
    const doc = { ...raw, execCommand: undefined } as unknown as LegacyCopyDocument;
    expect(legacyCopy("x", doc)).toBe(false);
    expect(raw.createElement).not.toHaveBeenCalled();
  });
});
