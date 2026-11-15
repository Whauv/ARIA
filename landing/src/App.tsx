import { MotionConfig } from "framer-motion";
import { AccentCursor } from "./components/AccentCursor";
import { AnnouncerProvider } from "./components/Announcer";
import { Footer } from "./components/Footer";
import { GestureReference } from "./components/GestureReference";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { Limitations } from "./components/Limitations";
import { Privacy } from "./components/Privacy";
import { RunLocally } from "./components/RunLocally";
import { Voice } from "./components/Voice";
import { Walkthrough } from "./components/Walkthrough";

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <AnnouncerProvider>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <Header />
        <main id="main" tabIndex={-1}>
          <Hero />
          <Walkthrough />
          <GestureReference />
          <Voice />
          <Privacy />
          <RunLocally />
          <Limitations />
        </main>
        <Footer />
        <AccentCursor />
      </AnnouncerProvider>
    </MotionConfig>
  );
}
