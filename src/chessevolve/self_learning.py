"""Verified recursive self-learning loop for ChessEvolve (Milestone 3).

Implements candidate training position selection, bounded Stockfish feedback
generation, strict example verification, candidate fine-tuning, and a
development-set acceptance gate with rollback.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .data import PositionRecord
from .dataset import TrainingExample, build_examples
from .encoding import encode_move, encode_position
from .engine import EngineSettings, StockfishAdapter
from .model import MLPRanker
from .distillation import _record_to_example
from .train import TrainConfig, evaluate, load_checkpoint, save_checkpoint, train_one_epoch


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InsufficientVerifiedExamplesError(ValueError):
    """Raised when fewer than the required verified examples remain."""


# --- 1. Position Selection --------------------------------------------------

@dataclass
class PositionSelectionConfig:
    policy: str = "uncertainty"  # "uncertainty", "random", or "disagreement"
    max_positions: int = 10
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def select_candidate_positions(
    records: List[PositionRecord],
    model: Optional[MLPRanker] = None,
    config: Optional[PositionSelectionConfig] = None,
) -> List[PositionRecord]:
    """Select candidate positions for external feedback from the training split.

    Parameters
    ----------
    records:
        List of position records. Must belong strictly to the "train" split.
    model:
        Current baseline model, required for "uncertainty" and "disagreement"
        policies.
    config:
        Selection configuration (policy, max_positions, seed).

    Returns
    -------
    List[PositionRecord]
        Selected training positions, up to ``max_positions``.

    Raises
    ------
    ValueError
        If test-split records are supplied or if policy requires a model that
        was not provided.
    """
    if config is None:
        config = PositionSelectionConfig()

    # Guard: input pool must contain only eligible training records.
    for r in records:
        if r.split != "train":
            raise ValueError(
                f"Non-training record {r.position_id} (split={r.split!r}) "
                f"must not enter candidate position selection."
            )

    train_records = list(records)
    if not train_records:
        return []

    if config.policy == "random":
        rng = random.Random(config.seed)
        k = min(len(train_records), config.max_positions)
        return rng.sample(train_records, k)

    if config.policy in {"uncertainty", "disagreement"}:
        if model is None:
            raise ValueError(f"Policy {config.policy!r} requires a trained model.")

        scored_records: List[Tuple[float, PositionRecord]] = []
        model.eval()
        with torch.no_grad():
            for rec in train_records:
                board = chess.Board(rec.fen)
                if not board.is_valid():
                    continue
                valid_moves = [
                    chess.Move.from_uci(m)
                    for m in rec.candidate_moves
                    if chess.Move.from_uci(m) in board.legal_moves
                ]
                if not valid_moves:
                    continue

                pos_feat = encode_position(board)
                move_feat = np.stack([encode_move(board, m) for m in valid_moves], axis=0)

                pos_t = torch.from_numpy(pos_feat).float()
                moves_t = torch.from_numpy(move_feat).float()
                logits = model(pos_t, moves_t)
                probs = F.softmax(logits, dim=-1)

                if config.policy == "uncertainty":
                    # Low margin between top 2 candidate moves indicates high uncertainty.
                    if len(valid_moves) >= 2:
                        top2 = torch.topk(probs, 2).values
                        margin = float((top2[0] - top2[1]).item())
                    else:
                        margin = 1.0
                    scored_records.append((margin, rec))

                elif config.policy == "disagreement":
                    # Prioritize positions where model disagrees with reference move, if known.
                    pred_idx = int(torch.argmax(logits).item())
                    pred_uci = valid_moves[pred_idx].uci()
                    ref_uci = rec.reference_best_move
                    is_disagreement = 1.0 if (ref_uci and pred_uci != ref_uci) else 0.0
                    # Higher disagreement priority (represented by negative score)
                    scored_records.append((-is_disagreement, rec))

        # Sort: lowest margin / highest disagreement first; break ties by position_id.
        scored_records.sort(key=lambda item: (item[0], item[1].position_id))
        selected = [rec for _, rec in scored_records[: config.max_positions]]
        return selected

    raise ValueError(f"Unknown selection policy: {config.policy!r}")


# --- 2. Feedback Generation -------------------------------------------------

@dataclass
class FeedbackRecord:
    """External feedback on a position with candidate moves and centipawn scores."""

    position_id: str
    source_game_id: str
    fen: str
    candidate_moves: List[str]
    candidate_scores: List[Optional[int]]
    split: str
    engine_version: str
    engine_settings: dict
    reference_best_move: Optional[str] = None
    is_valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def generate_feedback(
    records: List[PositionRecord],
    adapter: StockfishAdapter,
    settings: EngineSettings,
) -> List[FeedbackRecord]:
    """Generate bounded Stockfish evaluations for a list of position records.

    Catches and records engine failures gracefully without crashing.
    """
    feedback_list: List[FeedbackRecord] = []
    try:
        engine_ver = adapter.version()
    except Exception:
        engine_ver = "unknown"

    engine_settings_dict = {
        "depth": settings.depth,
        "time_seconds": settings.time_seconds,
    }

    for rec in records:
        board = chess.Board(rec.fen)
        if not board.is_valid():
            feedback_list.append(
                FeedbackRecord(
                    position_id=rec.position_id,
                    source_game_id=rec.source_game_id,
                    fen=rec.fen,
                    candidate_moves=rec.candidate_moves,
                    candidate_scores=[],
                    split=rec.split,
                    engine_version=engine_ver,
                    engine_settings=engine_settings_dict,
                    is_valid=False,
                    error="invalid_board_position",
                )
            )
            continue

        try:
            candidates = [chess.Move.from_uci(m) for m in rec.candidate_moves]
        except (chess.InvalidMoveError, chess.IllegalMoveError) as e:
            feedback_list.append(
                FeedbackRecord(
                    position_id=rec.position_id,
                    source_game_id=rec.source_game_id,
                    fen=rec.fen,
                    candidate_moves=rec.candidate_moves,
                    candidate_scores=[],
                    split=rec.split,
                    engine_version=engine_ver,
                    engine_settings=engine_settings_dict,
                    is_valid=False,
                    error=f"invalid_uci_move: {e}",
                )
            )
            continue

        if not candidates or any(m not in board.legal_moves for m in candidates):
            feedback_list.append(
                FeedbackRecord(
                    position_id=rec.position_id,
                    source_game_id=rec.source_game_id,
                    fen=rec.fen,
                    candidate_moves=rec.candidate_moves,
                    candidate_scores=[],
                    split=rec.split,
                    engine_version=engine_ver,
                    engine_settings=engine_settings_dict,
                    is_valid=False,
                    error="illegal_or_empty_candidates",
                )
            )
            continue

        try:
            scores = adapter.evaluate_candidates(board, candidates, settings)
        except Exception as exc:
            feedback_list.append(
                FeedbackRecord(
                    position_id=rec.position_id,
                    source_game_id=rec.source_game_id,
                    fen=rec.fen,
                    candidate_moves=rec.candidate_moves,
                    candidate_scores=[],
                    split=rec.split,
                    engine_version=engine_ver,
                    engine_settings=engine_settings_dict,
                    is_valid=False,
                    error=f"engine_error: {exc}",
                )
            )
            continue

        best_idx = None
        best_uci = None
        if scores and not any(s is None for s in scores):
            best_idx = max(range(len(scores)), key=lambda i: scores[i])  # type: ignore[arg-type]
            best_uci = candidates[best_idx].uci()

        feedback_list.append(
            FeedbackRecord(
                position_id=rec.position_id,
                source_game_id=rec.source_game_id,
                fen=rec.fen,
                candidate_moves=rec.candidate_moves,
                candidate_scores=scores,
                split=rec.split,
                engine_version=engine_ver,
                engine_settings=engine_settings_dict,
                reference_best_move=best_uci,
                is_valid=True,
            )
        )

    return feedback_list


# --- 3. Example Verifier ----------------------------------------------------

@dataclass
class VerifierConfig:
    allowed_split: str = "train"
    require_all_scores: bool = True
    min_candidates: int = 1
    max_candidates: Optional[int] = None
    min_score_margin_cp: Optional[int] = None  # optional score difference between top 2 moves

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    accepted_examples: List[TrainingExample]
    rejected_items: List[Tuple[FeedbackRecord, str]]
    total_reviewed: int
    num_accepted: int
    num_rejected: int
    rejection_reasons: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "total_reviewed": self.total_reviewed,
            "num_accepted": self.num_accepted,
            "num_rejected": self.num_rejected,
            "rejection_reasons": self.rejection_reasons,
        }


def verify_feedback(
    feedback_records: List[FeedbackRecord],
    config: Optional[VerifierConfig] = None,
    expected_records: Optional[Dict[str, PositionRecord]] = None,
) -> VerificationResult:
    """Verify generated feedback examples and reject invalid or untrusted data.

    Returns a structured :class:`VerificationResult` detailing accepted
    training examples and granular rejection reasons.
    """
    if config is None:
        config = VerifierConfig()

    accepted: List[TrainingExample] = []
    rejected: List[Tuple[FeedbackRecord, str]] = []
    reasons: Dict[str, int] = {}

    def _reject(rec: FeedbackRecord, reason: str) -> None:
        rejected.append((rec, reason))
        reasons[reason] = reasons.get(reason, 0) + 1

    for rec in feedback_records:
        if expected_records is not None:
            expected = expected_records.get(rec.position_id)
            if expected is None:
                _reject(rec, "unknown_position")
                continue
            if (
                rec.source_game_id != expected.source_game_id
                or rec.fen != expected.fen
                or rec.candidate_moves != expected.candidate_moves
                or rec.split != expected.split
            ):
                _reject(rec, "feedback_provenance_mismatch")
                continue

        # 1. Split verification
        if rec.split != config.allowed_split:
            _reject(rec, f"disallowed_split:{rec.split}")
            continue

        # 2. Engine error check
        if not rec.is_valid or rec.error:
            _reject(rec, f"engine_failure:{rec.error or 'unknown'}")
            continue

        # 3. Position board validity
        try:
            board = chess.Board(rec.fen)
        except Exception as exc:
            _reject(rec, f"malformed_fen:{exc}")
            continue

        if not board.is_valid():
            _reject(rec, "invalid_board_position")
            continue

        # 4. Candidates legality & count
        if not rec.candidate_moves:
            _reject(rec, "empty_candidates")
            continue

        if len(rec.candidate_moves) < config.min_candidates:
            _reject(rec, f"too_few_candidates:{len(rec.candidate_moves)}<{config.min_candidates}")
            continue

        if config.max_candidates and len(rec.candidate_moves) > config.max_candidates:
            _reject(rec, f"too_many_candidates:{len(rec.candidate_moves)}>{config.max_candidates}")
            continue

        moves: List[chess.Move] = []
        illegal_found = False
        for m_str in rec.candidate_moves:
            try:
                m = chess.Move.from_uci(m_str)
            except Exception:
                _reject(rec, f"malformed_uci_move:{m_str}")
                illegal_found = True
                break
            if m not in board.legal_moves:
                _reject(rec, f"illegal_move:{m_str}")
                illegal_found = True
                break
            moves.append(m)

        if illegal_found:
            continue

        # 5. Score validity and perspective consistency
        if len(rec.candidate_scores) != len(rec.candidate_moves):
            _reject(rec, "candidate_score_count_mismatch")
            continue

        if config.require_all_scores and any(s is None for s in rec.candidate_scores):
            _reject(rec, "missing_candidate_score")
            continue

        non_finite = any(
            s is None or not math.isfinite(s) or isinstance(s, bool)
            for s in rec.candidate_scores
        )
        if non_finite:
            _reject(rec, "non_finite_or_boolean_score")
            continue

        int_scores = [int(s) for s in rec.candidate_scores]  # type: ignore[arg-type]

        # 6. Optional score margin filter
        if config.min_score_margin_cp is not None and len(int_scores) >= 2:
            sorted_scores = sorted(int_scores, reverse=True)
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < config.min_score_margin_cp:
                _reject(rec, f"insufficient_score_margin:{margin}<{config.min_score_margin_cp}")
                continue

        best_index = int(max(range(len(int_scores)), key=lambda i: int_scores[i]))
        pos_feat = encode_position(board)
        move_feat = np.stack([encode_move(board, m) for m in moves], axis=0)

        evaluated_record = PositionRecord(
            position_id=rec.position_id,
            source_game_id=rec.source_game_id,
            fen=rec.fen,
            candidate_moves=rec.candidate_moves,
            split=rec.split,
            engine_version=rec.engine_version,
            engine_settings=rec.engine_settings,
            reference_best_move=moves[best_index].uci(),
            candidate_scores_cp=int_scores,
        )

        example = TrainingExample(
            position_features=pos_feat,
            move_features=move_feat,
            candidate_scores=int_scores,
            target_index=best_index,
            position_id=rec.position_id,
            fen=rec.fen,
            record=evaluated_record,
        )
        accepted.append(example)

    return VerificationResult(
        accepted_examples=accepted,
        rejected_items=rejected,
        total_reviewed=len(feedback_records),
        num_accepted=len(accepted),
        num_rejected=len(rejected),
        rejection_reasons=reasons,
    )


# --- 4. Candidate Fine-Tuning -----------------------------------------------

@dataclass
class FineTuneConfig:
    epochs: int = 2
    lr: float = 5e-4
    batch_size: int = 4
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def fine_tune_candidate(
    baseline_model: MLPRanker,
    examples: List[TrainingExample],
    config: Optional[FineTuneConfig] = None,
) -> MLPRanker:
    """Fine-tune a separate candidate copy of the baseline model.

    Leaves the baseline model completely unmodified.
    """
    if config is None:
        config = FineTuneConfig()

    candidate_model = MLPRanker(baseline_model.config)
    candidate_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))

    if not examples:
        return candidate_model

    optimizer = torch.optim.Adam(candidate_model.parameters(), lr=config.lr)
    # Seed both Python RNG and PyTorch RNG for reproducibility
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    for _ in range(config.epochs):
        train_one_epoch(candidate_model, examples, optimizer, config.batch_size, rng=rng)

    return candidate_model


# --- 5. Acceptance Gate -----------------------------------------------------

@dataclass
class GateConfig:
    min_regret_reduction_cp: float = 0.0  # candidate regret must be <= baseline - min_reduction
    max_agreement_drop: float = 0.05       # permitted drop in top-1 agreement
    require_finite_metrics: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    baseline_metrics: dict
    candidate_metrics: dict
    regret_diff_cp: float      # baseline regret - candidate regret (> 0 is improvement)
    agreement_diff: float      # candidate agreement - baseline agreement (> 0 is improvement)
    thresholds: dict

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_acceptance_gate(
    baseline_metrics: dict,
    candidate_metrics: dict,
    config: Optional[GateConfig] = None,
) -> GateDecision:
    """Evaluate candidate model against baseline model on development metrics.

    Decision policy:
    1. Metrics must be valid and finite.
    2. Candidate mean engine-score regret must improve by at least
       ``min_regret_reduction_cp`` relative to baseline.
    3. Candidate top-1 agreement drop must not exceed ``max_agreement_drop``.
    """
    if config is None:
        config = GateConfig()

    thresholds = config.to_dict()

    # Check finite metrics - always enforced, cannot be bypassed
    for name, m in [("baseline", baseline_metrics), ("candidate", candidate_metrics)]:
        if not isinstance(m, dict) or not m:
            return GateDecision(
                accepted=False,
                reason=f"{name} metrics missing or empty",
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                regret_diff_cp=0.0,
                agreement_diff=0.0,
                thresholds=thresholds,
            )
        # Check that we have at least some examples for a meaningful evaluation
        num_examples = m.get("num_examples", 0)
        if isinstance(num_examples, int) and num_examples == 0:
            return GateDecision(
                accepted=False,
                reason=f"{name} metrics based on 0 examples, cannot evaluate gate",
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                regret_diff_cp=0.0,
                agreement_diff=0.0,
                thresholds=thresholds,
            )
        required_metrics = ("loss", "top1_agreement", "mean_regret_cp")
        for k in required_metrics:
            v = m.get(k)
            if v is None or isinstance(v, bool) or not math.isfinite(float(v)):
                return GateDecision(
                    accepted=False,
                    reason=f"Non-finite or missing {name} metric {k}={v!r}",
                    baseline_metrics=baseline_metrics,
                    candidate_metrics=candidate_metrics,
                    regret_diff_cp=0.0,
                    agreement_diff=0.0,
                    thresholds=thresholds,
                )

    base_regret = baseline_metrics["mean_regret_cp"]
    cand_regret = candidate_metrics["mean_regret_cp"]
    regret_diff = base_regret - cand_regret  # positive means candidate regret is lower (better)

    base_top1 = baseline_metrics["top1_agreement"]
    cand_top1 = candidate_metrics["top1_agreement"]
    agreement_diff = cand_top1 - base_top1

    # Primary criterion: regret reduction
    if regret_diff < config.min_regret_reduction_cp:
        reason = (
            f"Regret reduction {regret_diff:.2f} cp did not meet threshold "
            f"{config.min_regret_reduction_cp:.2f} cp "
            f"(baseline={base_regret:.2f}, candidate={cand_regret:.2f})"
        )
        return GateDecision(
            accepted=False,
            reason=reason,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            regret_diff_cp=regret_diff,
            agreement_diff=agreement_diff,
            thresholds=thresholds,
        )

    # Secondary criterion: top-1 agreement degradation threshold
    agreement_drop = -agreement_diff
    if agreement_drop > config.max_agreement_drop:
        reason = (
            f"Top-1 agreement drop {agreement_drop:.4f} exceeded permitted threshold "
            f"{config.max_agreement_drop:.4f} "
            f"(baseline={base_top1:.4f}, candidate={cand_top1:.4f})"
        )
        return GateDecision(
            accepted=False,
            reason=reason,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            regret_diff_cp=regret_diff,
            agreement_diff=agreement_diff,
            thresholds=thresholds,
        )

    return GateDecision(
        accepted=True,
        reason="All acceptance gate criteria satisfied",
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        regret_diff_cp=regret_diff,
        agreement_diff=agreement_diff,
        thresholds=thresholds,
    )


# --- 6. Self-Learning Loop Orchestration -----------------------------------

@dataclass
class SelfLearningRoundConfig:
    seed: int = 42
    round_number: int = 1
    num_candidate_positions: int = 8
    selection_policy: str = "uncertainty"
    engine_depth: int = 4
    stockfish_path: str = "tools/stockfish/stockfish/stockfish-linux-x86-64-universal"
    fine_tune_epochs: int = 2
    fine_tune_lr: float = 5e-4
    fine_tune_batch_size: int = 4
    min_verified_examples: int = 2
    gate: GateConfig = field(default_factory=GateConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SelfLearningRoundResult:
    round_number: int
    decision: GateDecision
    accepted_checkpoint_path: Optional[str]
    retained_checkpoint_path: str
    verification_result: VerificationResult
    output_dir: str
    summary: dict

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "decision": self.decision.to_dict(),
            "accepted_checkpoint_path": self.accepted_checkpoint_path,
            "retained_checkpoint_path": self.retained_checkpoint_path,
            "verification_result": self.verification_result.to_dict(),
            "output_dir": self.output_dir,
            "summary": self.summary,
        }


@dataclass
class RecursiveSelfLearningConfig:
    """Configuration for a bounded sequence of verified improvement rounds."""

    iterations: int = 2
    round_config: SelfLearningRoundConfig = field(
        default_factory=SelfLearningRoundConfig
    )

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecursiveSelfLearningResult:
    """Audit result for all recursive rounds and the retained final checkpoint."""

    rounds: List[SelfLearningRoundResult]
    final_checkpoint_path: str
    summary: dict


def run_recursive_self_learning(
    baseline_checkpoint_path: Path,
    train_pool: List[PositionRecord],
    dev_pool: List[PositionRecord],
    output_dir: Path,
    config: Optional[RecursiveSelfLearningConfig] = None,
    dev_examples: Optional[List[TrainingExample]] = None,
) -> RecursiveSelfLearningResult:
    """Run several rounds, always continuing from the accepted checkpoint.

    Feedback is cached by position and engine settings, so a position selected
    again is not re-labeled.  The final-test split is intentionally not an
    input to this function; callers should evaluate it only after all rounds
    and gate settings are frozen.
    """
    if config is None:
        config = RecursiveSelfLearningConfig()
    if not train_pool or not dev_pool:
        raise ValueError("recursive self-learning requires non-empty train and dev pools")
    if any(record.split != "train" for record in train_pool):
        raise ValueError("recursive training pool must contain only train records")
    if any(record.split != "dev" for record in dev_pool):
        raise ValueError("recursive development pool must contain only dev records")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_checkpoint = Path(baseline_checkpoint_path)
    feedback_cache: Dict[str, FeedbackRecord] = {}
    for record in train_pool:
        if record.candidate_scores_cp is not None and record.engine_settings is not None:
            feedback_cache[record.position_id] = FeedbackRecord(
                position_id=record.position_id,
                source_game_id=record.source_game_id,
                fen=record.fen,
                candidate_moves=list(record.candidate_moves),
                candidate_scores=list(record.candidate_scores_cp),
                split=record.split,
                engine_version=record.engine_version or "unknown",
                engine_settings=dict(record.engine_settings),
                reference_best_move=record.reference_best_move,
            )
    round_results: List[SelfLearningRoundResult] = []

    for iteration in range(1, config.iterations + 1):
        round_config = dataclasses.replace(
            config.round_config,
            round_number=iteration,
            seed=config.round_config.seed + iteration - 1,
        )
        round_output = output_dir / f"round_{iteration:03d}"
        result = run_self_learning_round(
            current_checkpoint,
            train_pool,
            dev_pool,
            round_output,
            round_config,
            dev_examples=dev_examples,
            feedback_cache=feedback_cache,
        )
        round_results.append(result)
        current_checkpoint = round_output / "current_best.pt"

    summary = {
        "iterations": config.iterations,
        "rounds": [result.summary for result in round_results],
        "accepted_updates": sum(result.decision.accepted for result in round_results),
        "rejected_updates": sum(not result.decision.accepted for result in round_results),
        "final_checkpoint": str(current_checkpoint),
        "final_checkpoint_sha256": _checkpoint_sha256(current_checkpoint),
    }
    (output_dir / "recursive_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "recursive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return RecursiveSelfLearningResult(
        rounds=round_results,
        final_checkpoint_path=str(current_checkpoint),
        summary=summary,
    )


def run_self_learning_round(
    baseline_checkpoint_path: Path,
    train_pool: List[PositionRecord],
    dev_pool: List[PositionRecord],
    output_dir: Path,
    config: Optional[SelfLearningRoundConfig] = None,
    dev_examples: Optional[List[TrainingExample]] = None,
    feedback_cache: Optional[Dict[str, FeedbackRecord]] = None,
) -> SelfLearningRoundResult:
    """Execute one complete round of the verified self-learning loop.

    Steps:
    1. Load baseline checkpoint safely without in-place modification.
    2. Select candidate positions from ``train_pool`` using declared policy.
    3. Generate bounded Stockfish feedback with an isolated engine session.
    4. Verify feedback; reject untrusted examples.
    5. Fine-tune a candidate model on verified examples.
    6. Evaluate baseline and candidate on identical development positions.
    7. Make gate decision (ACCEPT / REJECT).
    8. Save structured logs, metrics, decisions, and checkpoints.
    """
    if config is None:
        config = SelfLearningRoundConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale checkpoint files from previous rejected rounds
    # to prevent reuse of accepted.pt after later rejection
    for stale in output_dir.glob("accepted.pt"):
        stale.unlink()
    for stale in output_dir.glob("candidate.pt"):
        stale.unlink()

    # 0. Strict split isolation guard
    for r in train_pool:
        if r.split == "test":
            raise ValueError(
                f"Test-set record {r.position_id} detected in training pool: strict isolation violated."
            )
    for r in dev_pool:
        if r.split == "test":
            raise ValueError(
                f"Test-set record {r.position_id} detected in dev pool: strict isolation violated."
            )

    # 0a. Split identity guard: train_pool must contain only train records, dev_pool only dev
    train_splits = {r.split for r in train_pool}
    dev_splits = {r.split for r in dev_pool}
    if "train" in dev_splits:
        raise ValueError(
            f"train-labeled records found in dev_pool: { [r.position_id for r in dev_pool if r.split == 'train'] }"
        )
    if "dev" in train_splits:
        raise ValueError(
            f"dev-labeled records found in train_pool: { [r.position_id for r in train_pool if r.split == 'dev'] }"
        )

    # 0b. Source-game overlap guard: no record should appear in both train and dev pools
    train_game_ids = {r.source_game_id for r in train_pool}
    dev_game_ids = {r.source_game_id for r in dev_pool}
    overlap = train_game_ids & dev_game_ids
    if overlap:
        raise ValueError(
            f"Source-game ID(s) {overlap} appear in both train and dev pools; splits must be disjoint."
        )

    # 1. Load baseline model
    baseline_model, baseline_ckpt = load_checkpoint(baseline_checkpoint_path)
    # Reconstruct TrainConfig from saved dict so save_checkpoint() can call .to_dict().
    baseline_train_config = TrainConfig(**baseline_ckpt["train_config"])

    # 2. Select candidate positions
    sel_config = PositionSelectionConfig(
        policy=config.selection_policy,
        max_positions=config.num_candidate_positions,
        seed=config.seed,
    )
    selected_records = select_candidate_positions(train_pool, model=baseline_model, config=sel_config)

    # 3. Generate Stockfish feedback
    engine_settings = EngineSettings(depth=config.engine_depth)
    selected_by_id = {record.position_id: record for record in selected_records}
    settings_key = (engine_settings.depth, engine_settings.time_seconds)
    cached_feedback: List[FeedbackRecord] = []
    uncached_records: List[PositionRecord] = []
    for record in selected_records:
        cached = feedback_cache.get(record.position_id) if feedback_cache is not None else None
        if (
            cached is not None
            and cached.engine_settings.get("depth") == settings_key[0]
            and cached.engine_settings.get("time_seconds") == settings_key[1]
        ):
            cached_feedback.append(cached)
        else:
            uncached_records.append(record)

    feedback_records = list(cached_feedback)
    if uncached_records:
        with StockfishAdapter(Path(config.stockfish_path)) as adapter:
            adapter.reset()
            generated = generate_feedback(uncached_records, adapter, engine_settings)
        feedback_records.extend(generated)
        if feedback_cache is not None:
            feedback_cache.update({item.position_id: item for item in generated})
    feedback_records.sort(key=lambda item: selected_by_id.get(item.position_id, selected_records[0]).position_id)

    # 4. Verify examples
    verification = verify_feedback(
        feedback_records,
        config.verifier,
        expected_records=selected_by_id,
    )
    if verification.num_accepted < config.min_verified_examples:
        # Save rejection metadata and halt
        summary = {
            "status": "failed",
            "reason": (
                f"Insufficient verified examples: got {verification.num_accepted}, "
                f"required {config.min_verified_examples}"
            ),
            "verification": verification.to_dict(),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        raise InsufficientVerifiedExamplesError(summary["reason"])

    # 5. Fine-tune candidate model
    ft_config = FineTuneConfig(
        epochs=config.fine_tune_epochs,
        lr=config.fine_tune_lr,
        batch_size=config.fine_tune_batch_size,
        seed=config.seed,
    )
    candidate_model = fine_tune_candidate(baseline_model, verification.accepted_examples, ft_config)

    # Save candidate checkpoint before gate decision
    candidate_ckpt_path = output_dir / "candidate.pt"
    save_checkpoint(
        candidate_ckpt_path,
        candidate_model,
        baseline_train_config,
        {"round": config.round_number, "verified_examples": verification.num_accepted},
        {"role": "candidate", "round": config.round_number},
    )

    # 6. Evaluate baseline and candidate on identical dev positions
    if dev_examples is None:
        if all(record.candidate_scores_cp is not None for record in dev_pool):
            dev_examples = [_record_to_example(record) for record in dev_pool]
        else:
            with StockfishAdapter(Path(config.stockfish_path)) as dev_adapter:
                dev_adapter.reset()
                dev_examples, _ = build_examples(dev_pool, dev_adapter, engine_settings)

    baseline_metrics = evaluate(baseline_model, dev_examples)
    candidate_metrics = evaluate(candidate_model, dev_examples)

    # 7. Gate decision
    decision = evaluate_acceptance_gate(baseline_metrics, candidate_metrics, config.gate)

    # 8. Checkpoint management and artifacts
    retained_ckpt_path = output_dir / "retained.pt"
    save_checkpoint(
        retained_ckpt_path,
        baseline_model,
        baseline_train_config,
        baseline_ckpt.get("split_info", {}),
        baseline_metrics,
    )

    accepted_ckpt_path: Optional[str] = None
    if decision.accepted:
        accepted_path = output_dir / "accepted.pt"
        save_checkpoint(
            accepted_path,
            candidate_model,
            baseline_train_config,
            {"round": config.round_number, "verified_examples": verification.num_accepted},
            candidate_metrics,
        )
        accepted_ckpt_path = str(accepted_path)
        current_best_path = output_dir / "current_best.pt"
        save_checkpoint(
            current_best_path,
            candidate_model,
            baseline_train_config,
            {"round": config.round_number, "decision": "accepted"},
            candidate_metrics,
        )
    else:
        current_best_path = output_dir / "current_best.pt"
        save_checkpoint(
            current_best_path,
            baseline_model,
            baseline_train_config,
            {"round": config.round_number, "decision": "retained"},
            baseline_metrics,
        )

    # Write structured audit artifacts
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (output_dir / "decision.json").write_text(json.dumps(decision.to_dict(), indent=2))
    (output_dir / "metrics_baseline.json").write_text(json.dumps(baseline_metrics, indent=2))
    (output_dir / "metrics_candidate.json").write_text(json.dumps(candidate_metrics, indent=2))
    (output_dir / "rejection_summary.json").write_text(json.dumps(verification.rejection_reasons, indent=2))

    with (output_dir / "feedback_records.jsonl").open("w", encoding="utf-8") as handle:
        for f in feedback_records:
            handle.write(json.dumps(f.to_dict()) + "\n")

    with (output_dir / "rejections.jsonl").open("w", encoding="utf-8") as handle:
        for item, reason in verification.rejected_items:
            handle.write(json.dumps({"position_id": item.position_id, "reason": reason}) + "\n")

    summary = {
        "round_number": config.round_number,
        "seed": config.seed,
        "accepted": decision.accepted,
        "decision_reason": decision.reason,
        "regret_diff_cp": decision.regret_diff_cp,
        "agreement_diff": decision.agreement_diff,
        "num_candidate_positions": len(selected_records),
        "num_feedback_generated": len(feedback_records),
        "num_verified_examples": verification.num_accepted,
        "num_rejected_examples": verification.num_rejected,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "checkpoint_hashes": {
            "baseline_input": _checkpoint_sha256(baseline_checkpoint_path),
            "candidate": _checkpoint_sha256(candidate_ckpt_path),
            "retained": _checkpoint_sha256(retained_ckpt_path),
            "current_best": _checkpoint_sha256(current_best_path),
        },
    }
    (output_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))

    return SelfLearningRoundResult(
        round_number=config.round_number,
        decision=decision,
        accepted_checkpoint_path=accepted_ckpt_path,
        retained_checkpoint_path=str(retained_ckpt_path),
        verification_result=verification,
        output_dir=str(output_dir),
        summary=summary,
    )
