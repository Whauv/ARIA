import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

type Announce = (message: string) => void;

const AnnouncerContext = createContext<Announce>(() => {});

/** Single polite live region shared by the page (e.g. "Copied"). */
export function AnnouncerProvider({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState("");
  const timer = useRef<number | undefined>(undefined);

  const announce = useCallback<Announce>((next) => {
    // Clear first so repeating the same message is announced again.
    setMessage("");
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setMessage(next), 60);
  }, []);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  return (
    <AnnouncerContext.Provider value={announce}>
      {children}
      <div className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
        {message}
      </div>
    </AnnouncerContext.Provider>
  );
}

export function useAnnounce(): Announce {
  return useContext(AnnouncerContext);
}
