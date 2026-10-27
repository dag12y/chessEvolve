export const MAX_PLIES = 200;

export function parseUci(
  uci: string
): { from: string; to: string; promotion?: "q" | "r" | "b" | "n" } {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    ...(uci.length === 5
      ? { promotion: uci[4] as "q" | "r" | "b" | "n" }
      : {}),
  };
}

export function sideName(color: "w" | "b" | null): string {
  if (color === "w") return "White";
  if (color === "b") return "Black";
  return "—";
}
