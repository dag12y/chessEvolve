/// <reference types="vite/client" />

export const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export interface Model {
  id: string;
  name: string;
  type: "checkpoint" | "random";
  path?: string;
  available: boolean;
}

export interface TopMove {
  rank: number;
  uci: string;
  score: number | null;
  probability: number | null;
}

export interface MoveResponse {
  model_id: string;
  fen: string;
  temperature: number;
  chosen: { uci: string; san: string; score: number | null };
  top_moves: TopMove[];
  legal_moves: number;
  position_after: {
    fen: string;
    check: boolean;
    game_over: boolean;
    result: string | null;
  };
}

async function request(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {}
    throw new Error(`${path} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function listModels(): Promise<Model[]> {
  const data = await request("/models");
  return data.models;
}

export async function getMove(
  fen: string,
  modelId: string,
  temperature: number,
  searchDepth = 2
): Promise<MoveResponse> {
  return request("/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, model_id: modelId, temperature, search_depth: searchDepth }),
  });
}
