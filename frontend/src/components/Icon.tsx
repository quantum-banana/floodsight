import type { ReactNode } from "react";

interface IconProps {
  name:
    | "activity"
    | "alert"
    | "building"
    | "check"
    | "chevron"
    | "clipboard"
    | "close"
    | "diagnostics"
    | "eye"
    | "focus"
    | "layers"
    | "map"
    | "pause"
    | "people"
    | "play"
    | "print"
    | "report"
    | "reset"
    | "road"
    | "route"
    | "vehicle"
    | "water";
  className?: string;
}

const paths: Record<IconProps["name"], ReactNode> = {
  activity: <path d="M3 12h4l2-6 4 12 2-6h6" />,
  alert: <path d="M12 9v4m0 4h.01M10.3 4.3 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0Z" />,
  building: <path d="M4 21V5l8-3 8 3v16M8 8h2m4 0h2M8 12h2m4 0h2M8 16h2m4 0h2M2 21h20" />,
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m9 18 6-6-6-6" />,
  clipboard: <path d="M9 5h6m-7 3h8m-8 4h8m-8 4h5M9 3h6a2 2 0 0 1 2 2h2v16H5V5h2a2 2 0 0 1 2-2Z" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  diagnostics: <path d="M4 19V5m0 7h4l2-5 4 10 2-5h4M4 19h16" />,
  eye: <><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z" /><circle cx="12" cy="12" r="2.5" /></>,
  focus: <path d="M8 3H3v5m13-5h5v5M8 21H3v-5m13 5h5v-5M9 9h6v6H9z" />,
  layers: <path d="m12 3 9 5-9 5-9-5 9-5Zm-9 10 9 5 9-5M3 17l9 5 9-5" />,
  map: <path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3V6Zm6-3v15m6-12v15" />,
  pause: <path d="M8 5v14m8-14v14" />,
  people: <><circle cx="9" cy="8" r="3" /><circle cx="17" cy="9" r="2.5" /><path d="M3 21v-2a6 6 0 0 1 12 0v2m1-7a5 5 0 0 1 5 5v2" /></>,
  play: <path d="m8 5 11 7-11 7V5Z" />,
  print: <path d="M7 8V3h10v5M7 17H4a2 2 0 0 1-2-2v-5h20v5a2 2 0 0 1-2 2h-3m-10-4h10v8H7v-8Z" />,
  report: <path d="M6 3h9l4 4v14H6V3Zm8 0v5h5M9 12h6m-6 4h6" />,
  reset: <path d="M4 4v6h6M5.5 15a8 8 0 1 0 1.2-8.7L4 10" />,
  road: <path d="M8 3 5 21m11-18 3 18M12 4v3m0 4v3m0 4v3" />,
  route: <><circle cx="6" cy="18" r="2" /><circle cx="18" cy="6" r="2" /><path d="M8 18h3a3 3 0 0 0 3-3v-6a3 3 0 0 1 3-3" /></>,
  vehicle: <path d="m5 17-2-2v-4l2-5h14l2 5v4l-2 2M5 17v3m14-3v3M3 12h18M7 15h.01M17 15h.01" />,
  water: <path d="M3 8c3-2 6 2 9 0s6 2 9 0M3 13c3-2 6 2 9 0s6 2 9 0M3 18c3-2 6 2 9 0s6 2 9 0" />,
};

export function Icon({ name, className = "h-4 w-4" }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
    >
      {paths[name]}
    </svg>
  );
}
