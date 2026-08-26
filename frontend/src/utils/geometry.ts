import type { Point } from "../types/liveResult";

export const toSvgPoints = (points: Point[]): string =>
  points.map(({ x, y }) => `${x * 100},${y * 100}`).join(" ");

export const toSvgPath = (points: Point[]): string => {
  if (!points.length) return "";
  const [first, ...rest] = points;
  return `M ${first.x * 100} ${first.y * 100} ${rest
    .map(({ x, y }) => `L ${x * 100} ${y * 100}`)
    .join(" ")}`;
};
