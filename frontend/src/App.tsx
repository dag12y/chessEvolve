import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Chess, type Square } from "chess.js";
import GameBoard from "./components/GameBoard";
import MoveList from "./components/MoveList";
import GameStatus from "./components/GameStatus";
import ModeSelector from "./components/ModeSelector";
import ModelSelect from "./components/ModelSelect";
import AutoplayControls from "./components/AutoplayControls";
import {
  listModels,
  getMove,
  type Model,
  type TopMove,
} from "./lib/api";
import { MAX_PLIES, parseUci, sideName } from "./lib/game";
import "./index.css";

const DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
type Mode = "human" | "model";

interface Suggestion {
  rank: number;
  uci: string;
  san: string;
  score: number | null;
  probability: number | null;
}

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("human");
  const [humanSide, setHumanSide] = useState<"w" | "b">("w");
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [whiteModel, setWhiteModel] = useState<string | null>(null);
  const [blackModel, setBlackModel] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0);
  const [speed, setSpeed] = useState(600);

  const gameRef = useRef(new Chess(DEFAULT_FEN));
  const [position, setPosition] = useState(DEFAULT_FEN);
  const [moveList, setMoveList] = useState<string[]>([]);
  const [pendingPromotion, setPendingPromotion] = useState<{
    from: string;
    to: string;
  } | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [wouldPlay, setWouldPlay] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const positionRef = useRef(position);
  const moveRequestRef = useRef(false);
  const gameVersionRef = useRef(0);
  const isPlayingRef = useRef(isPlaying);

  positionRef.current = position;
  isPlayingRef.current = isPlaying;

  const modelById = useMemo(
    () => Object.fromEntries(models.map((m) => [m.id, m])),
    [models]
  );

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch((err) =>
        setModelsError(err instanceof Error ? err.message : String(err))
      );
  }, []);

  const availableModels = models.filter((m) => m.available);
  const gameOver = gameRef.current.isGameOver();
  const plies = moveList.length;
  const plyCapReached = mode === "model" && plies >= MAX_PLIES;

  const resetGame = (fen: string = DEFAULT_FEN) => {
    gameVersionRef.current += 1;
    moveRequestRef.current = false;
    gameRef.current = new Chess(fen);
    setPosition(fen);
    setMoveList([]);
    setPendingPromotion(null);
    setSuggestions([]);
    setWouldPlay(null);
    setError(null);
    setIsPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  // New game whenever mode or side changes.
  useEffect(() => {
    resetGame();
  }, [mode, humanSide]);

  const applyUci = useCallback((uci: string): boolean => {
    const g = gameRef.current;
    const { from, to, promotion } = parseUci(uci);
    try {
      const move = g.move({ from: from as Square, to: to as Square, promotion });
      if (!move) return false;
    } catch {
      return false;
    }
    setPosition(g.fen());
    setMoveList(g.history({ verbose: true }).map((m) => m.san));
    return true;
  }, []);

  // --- Human vs Model: model replies automatically -----------------------
  useEffect(() => {
    if (mode !== "human") return;
    const g = gameRef.current;
    if (g.isGameOver()) return;
    if (g.turn() !== "w" && g.turn() !== "b") return;
    if (g.turn() === humanSide) return;
    if (!selectedModel) {
      setError("Select a model to play against.");
      return;
    }
    if (moveRequestRef.current) return;
    timerRef.current = setTimeout(async () => {
      if (gameRef.current.isGameOver() || gameRef.current.turn() === humanSide) return;
      moveRequestRef.current = true;
      const requestVersion = gameVersionRef.current;
      try {
        const body = await getMove(positionRef.current, selectedModel, temperature);
        if (requestVersion !== gameVersionRef.current) return;
        if (!applyUci(body.chosen.uci)) {
          throw new Error("Model returned an illegal move.");
        }
        setSuggestions(backendTopMovesToSuggestions(body.top_moves));
        setWouldPlay(body.chosen.uci);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        moveRequestRef.current = false;
      }
    }, 500);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [position, mode, humanSide, selectedModel, temperature, applyUci]);

  // --- Model vs Model: autoplay loop -------------------------------------
  const playOne = useCallback(async () => {
    const g = gameRef.current;
    if (g.isGameOver() || moveRequestRef.current) {
      setIsPlaying(false);
      return;
    }
    if (g.history().length >= MAX_PLIES) {
      setIsPlaying(false);
      return;
    }
    const modelId = g.turn() === "w" ? whiteModel : blackModel;
    if (!modelId) {
      setIsPlaying(false);
      setError("Select both models.");
      return;
    }
    moveRequestRef.current = true;
    const requestVersion = gameVersionRef.current;
    try {
      const body = await getMove(positionRef.current, modelId, temperature);
      if (requestVersion !== gameVersionRef.current) return;
      if (!applyUci(body.chosen.uci)) {
        setIsPlaying(false);
        setError("Model returned an illegal move.");
        return;
      }
      setSuggestions(backendTopMovesToSuggestions(body.top_moves));
      setWouldPlay(body.chosen.uci);
      if (gameRef.current.isGameOver() || gameRef.current.history().length >= MAX_PLIES) {
        setIsPlaying(false);
        return;
      }
      if (isPlayingRef.current) {
        timerRef.current = setTimeout(playOne, speed);
      }
    } catch (err) {
      setIsPlaying(false);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      moveRequestRef.current = false;
    }
  }, [mode, temperature, whiteModel, blackModel, speed, applyUci]);

  useEffect(() => {
    if (!isPlaying || mode !== "model") return;
    timerRef.current = setTimeout(playOne, speed);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, mode, playOne, speed]);

  // --- Display-only suggestions for the current position -----------------
  const fetchSuggestions = useCallback(async () => {
    const g = gameRef.current;
    if (g.isGameOver() || moveRequestRef.current) return;
    if (mode === "human" && g.turn() !== humanSide) return;
    if (mode === "model" && isPlaying) return;
    const modelId =
      mode === "human"
        ? selectedModel
        : g.turn() === "w"
          ? whiteModel
          : blackModel;
    if (!modelId) return;
    try {
      const body = await getMove(positionRef.current, modelId, temperature);
      setSuggestions(backendTopMovesToSuggestions(body.top_moves));
      setWouldPlay(body.chosen.uci);
    } catch {
      // Suggestions are best-effort; silence failures.
    }
  }, [
    position,
    mode,
    humanSide,
    isPlaying,
    selectedModel,
    whiteModel,
    blackModel,
    temperature,
  ]);

  useEffect(() => {
    fetchSuggestions();
  }, [position, fetchSuggestions]);

  // --- Human board interaction -------------------------------------------
  const handleBoardMove = useCallback(
    (source: string, target: string): boolean => {
      const g = gameRef.current;
      if (g.isGameOver()) return false;
      if (mode === "human" && g.turn() !== humanSide) return false;
      const piece = g.get(source as Square);
      const isPromotion =
        piece &&
        piece.type === "p" &&
        ((piece.color === "w" && target[1] === "8") ||
          (piece.color === "b" && target[1] === "1"));
      if (isPromotion) {
        setPendingPromotion({ from: source, to: target });
        return false;
      }
      const move = g.move({ from: source as Square, to: target as Square });
      if (!move) return false;
      setPosition(g.fen());
      setMoveList(g.history({ verbose: true }).map((m) => m.san));
      return true;
    },
    [mode, humanSide]
  );

  const canDragPiece = useCallback(
    (args: {
      isSparePiece: boolean;
      piece: { pieceType: string };
      square: string | null;
    }) => {
      if (mode !== "human") return false;
      const g = gameRef.current;
      if (g.turn() !== humanSide) return false;
      if (!args.square) return false;
      const p = g.get(args.square as Square);
      return p ? p.color === humanSide : false;
    },
    [mode, humanSide]
  );

  const handlePromotionSelect = (piece: "q" | "r" | "b" | "n") => {
    const pending = pendingPromotion;
    if (!pending) return;
    const g = gameRef.current;
    const move = g.move({
      from: pending.from,
      to: pending.to,
      promotion: piece,
    });
    if (!move) return;
    setPosition(g.fen());
    setMoveList(g.history({ verbose: true }).map((m) => m.san));
    setPendingPromotion(null);
  };

  const handleNewGame = () => resetGame();
  const handleExportPgn = () => {
    const pgn = gameRef.current.pgn();
    const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "chessevolve-game.pgn";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const humanVsModel =
    mode === "human" &&
    !gameOver &&
    gameRef.current.turn() !== null &&
    gameRef.current.turn() !== humanSide;
  const turnLabel = (() => {
    const g = gameRef.current;
    if (g.isGameOver()) return "Game over";
    if (mode === "human")
      return g.turn() === humanSide
        ? "Your turn"
        : `${sideName(g.turn())} — ${
            modelById[selectedModel ?? ""]?.name ?? selectedModel ?? "model"
          }`;
    return `${sideName(g.turn())} to move`;
  })();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">♞</div>
          <div>
            <div className="brand-name">ChessEvolve</div>
            <div className="brand-subtitle">Neural chess laboratory</div>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="live-dot" /> Local CPU inference
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <div className="panel setup-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Game setup</span>
                <h1>Choose a matchup</h1>
              </div>
              <span className="step-badge">01</span>
            </div>
            <ModeSelector mode={mode} onChange={setMode} />
            <div className="control-stack">
              {mode === "human" ? (
                <>
                  <div className="field">
                    <label>Play as</label>
                    <select
                      value={humanSide}
                      onChange={(e) => setHumanSide(e.target.value as "w" | "b")}
                    >
                      <option value="w">White</option>
                      <option value="b">Black</option>
                    </select>
                  </div>
                  <ModelSelect
                    label="Opponent model"
                    value={selectedModel}
                    models={availableModels}
                    onChange={setSelectedModel}
                  />
                </>
              ) : (
                <>
                  <ModelSelect
                    label="White model"
                    value={whiteModel}
                    models={availableModels}
                    onChange={setWhiteModel}
                  />
                  <ModelSelect
                    label="Black model"
                    value={blackModel}
                    models={availableModels}
                    onChange={setBlackModel}
                  />
                </>
              )}
              <div className="field-inline">
                <label htmlFor="temperature">Temperature</label>
                <input
                  id="temperature"
                  type="number"
                  min="0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) =>
                    setTemperature(Math.max(0, parseFloat(e.target.value) || 0))
                  }
                />
              </div>
              <div className="helper-text">
                0 selects the highest-ranked move. Higher values add variation.
              </div>
            </div>
            {mode === "model" && (
              <div className="limit-note">
                <span>♟</span> Model games end after {MAX_PLIES} plies
              </div>
            )}
            <div className="setup-actions">
              <button type="button" onClick={handleNewGame} className="primary action-main">
                <span>＋</span> New game
              </button>
              <button
                type="button"
                onClick={handleExportPgn}
                disabled={moveList.length === 0}
                className="icon-button"
                title="Export PGN"
              >
                ↓
              </button>
            </div>
            {modelsError && <div className="err">Models unavailable: {modelsError}</div>}
            {error && <div className="err">{error}</div>}
          </div>

          <div className="panel moves-panel">
            <div className="panel-heading compact">
              <div>
                <span className="eyebrow">Notation</span>
                <h2>Move history</h2>
              </div>
              <span className="move-count">{moveList.length} ply</span>
            </div>
            <MoveList moves={moveList} />
          </div>
        </aside>

        <section className="game-area">
          <div className="game-header">
            <div>
              <span className="eyebrow">{mode === "human" ? "Human vs model" : "Model arena"}</span>
              <h2>{turnLabel}</h2>
            </div>
            <div className="game-actions">
              {humanVsModel && <span className="thinking-pill"><span className="pulse" /> Model thinking</span>}
              <button type="button" onClick={handleExportPgn} disabled={moveList.length === 0}>
                Export PGN
              </button>
            </div>
          </div>

          <div className="board-card">
            <div className="player-strip top-player">
              <span className="color-token dark-token">♚</span>
              <span>{mode === "model" ? (modelById[blackModel ?? ""]?.name ?? "Black model") : "Black"}</span>
              <span className="player-side">BLACK</span>
            </div>
            <GameBoard
              position={position}
              orientation={mode === "human" ? (humanSide === "w" ? "white" : "black") : "white"}
              allowDragging={mode === "human" && !gameOver && !pendingPromotion}
              canDragPiece={canDragPiece}
              onPieceDrop={handleBoardMove}
              onPromotionSelect={handlePromotionSelect}
              pendingPromotion={pendingPromotion}
            />
            <div className="player-strip bottom-player">
              <span className="color-token light-token">♔</span>
              <span>{mode === "model" ? (modelById[whiteModel ?? ""]?.name ?? "White model") : "White"}</span>
              <span className="player-side">WHITE</span>
            </div>
          </div>
        </section>

        <aside className="inspector">
          <div className="panel status-panel">
            <span className="eyebrow">Position</span>
            <GameStatus fen={position} plyCapReached={plyCapReached} />
          </div>
          {mode === "model" && (
            <div className="panel autoplay-panel">
              <div className="panel-heading compact">
                <div>
                  <span className="eyebrow">Simulation</span>
                  <h2>Playback</h2>
                </div>
                <span className={isPlaying ? "state-dot active" : "state-dot"} />
              </div>
              <AutoplayControls
                isPlaying={isPlaying}
                speed={speed}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onStep={playOne}
                onSpeedChange={setSpeed}
              />
            </div>
          )}
          <div className="panel analysis-panel">
            <div className="panel-heading compact">
              <div>
                <span className="eyebrow">Model analysis</span>
                <h2>Top candidates</h2>
              </div>
              <span className="analysis-side">{sideName(gameRef.current.turn())}</span>
            </div>
            <div className="analysis-model">
              {modelById[
                (mode === "human" ? selectedModel : gameRef.current.turn() === "w" ? whiteModel : blackModel) ?? ""
              ]?.name ?? "Select a model"}
            </div>
            {suggestions.length === 0 && !gameOver && (
              <div className="empty-analysis">Choose a model to see ranked moves.</div>
            )}
            <div className="suggestion-list">
              {suggestions.map((s) => (
                <div key={s.rank} className="suggestion-card">
                  <span className="rank-number">{String(s.rank).padStart(2, "0")}</span>
                  <span className="suggestion-move">{s.uci}</span>
                  <span className="suggestion-san">{s.san}</span>
                  <span className="suggestion-score">
                    {s.score === null ? "—" : s.score.toFixed(2)}
                    {s.probability !== null && <small>{(s.probability * 100).toFixed(1)}%</small>}
                  </span>
                </div>
              ))}
            </div>
            {wouldPlay && (
              <div className="would-play">
                <span>Recommended move</span>
                <strong>{wouldPlay}</strong>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

function backendTopMovesToSuggestions(topMoves: TopMove[]): Suggestion[] {
  return topMoves.map((m) => ({
    rank: m.rank,
    uci: m.uci,
    san: m.uci,
    score: m.score,
    probability: m.probability,
  }));
}
