import { useEffect, useRef } from "react";

export function WaterAmbience() {
  const waterRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = waterRef.current;
    if (!element) return;

    const reducedMotion = typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : null;
    if (reducedMotion?.matches) {
      element.style.setProperty("--water-progress", "0.32");
      return;
    }

    let animationFrame: number | null = null;
    const updateWaterLevel = () => {
      animationFrame = null;
      const scrollRange = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const progress = Math.min(1, Math.max(0, window.scrollY / scrollRange));
      element.style.setProperty("--water-progress", progress.toFixed(3));
    };
    const requestUpdate = () => {
      if (animationFrame === null) animationFrame = window.requestAnimationFrame(updateWaterLevel);
    };

    updateWaterLevel();
    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate, { passive: true });
    return () => {
      window.removeEventListener("scroll", requestUpdate);
      window.removeEventListener("resize", requestUpdate);
      if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return (
    <div ref={waterRef} className="water-ambience" aria-hidden="true">
      <svg viewBox="0 0 2400 300" preserveAspectRatio="none">
        <defs>
          <linearGradient id="arctic-water-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#8edfec" stopOpacity="0.09" />
            <stop offset="1" stopColor="#cceff6" stopOpacity="0.22" />
          </linearGradient>
          <linearGradient id="arctic-water-fill-soft" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#58c9dc" stopOpacity="0.05" />
            <stop offset="1" stopColor="#bce9f1" stopOpacity="0.14" />
          </linearGradient>
        </defs>
        <path
          className="water-wave water-wave-back"
          d="M0 76C160 36 320 108 480 76S800 36 960 76s320 40 480 0 320-40 480 0 320 40 480 0V300H0Z"
          fill="url(#arctic-water-fill-soft)"
        />
        <path
          className="water-wave water-wave-front"
          d="M0 94c200-46 280 34 480 0s280-46 480 0 280 34 480 0 280-46 480 0 280 34 480 0V300H0Z"
          fill="url(#arctic-water-fill)"
        />
      </svg>
    </div>
  );
}

export function WaveLoader() {
  return (
    <div className="wave-loader" aria-hidden="true">
      <svg viewBox="0 0 160 44" preserveAspectRatio="none">
        <path className="wave-loader-line wave-loader-line-back" d="M-20 24C5 8 25 40 50 24S95 8 120 24s45 16 70 0" />
        <path className="wave-loader-line wave-loader-line-front" d="M-30 29C-5 13 15 45 40 29s45-16 70 0 45 16 70 0 45-16 70 0" />
      </svg>
    </div>
  );
}
