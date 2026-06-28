"""Emit train/holdout JSONL and generate corpus audit report."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from datasets.corpus.curated import build_bias_records, build_fallacy_records
from datasets.corpus.dedup import (
    assert_holdout_disjoint,
    deduplicate_records,
    filter_phase8_overlap,
)
from datasets.corpus.ingest import ingest_all, load_corpus_config
from datasets.corpus.metadata import filter_valid_records, load_source_registry
from datasets.corpus.query import metadata_coverage, topic_tradition_heatmap
from datasets.corpus.schema import CorpusRecord, write_jsonl
from datasets.corpus.split import ChunkPolicy
from datasets.corpus.transcripts import ingest_all_transcripts


def load_phase8_reference_texts(shards_dir: Path, *, max_shards: int = 5) -> list[str]:
    """Load sample texts from Phase 8 general shards for overlap checking."""
    texts: list[str] = []
    train_dir = shards_dir / "train"
    if not train_dir.exists():
        return texts
    shard_paths = sorted(train_dir.glob("*.pt"))[:max_shards]
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        chunks = payload.get("chunks", [])
        for chunk in chunks[:20]:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, list):
                texts.append(" ".join(str(t) for t in chunk))
    return texts


def compute_mix_buckets(records: list[CorpusRecord]) -> dict[str, int]:
    """Sum token counts per mix bucket for train split."""
    totals: dict[str, int] = defaultdict(int)
    for record in records:
        if record.split != "train":
            continue
        bucket = record.mix_bucket or "other"
        totals[bucket] += record.token_count
    return dict(totals)


def generate_audit_report(
    records: list[CorpusRecord],
    config: dict[str, Any],
    *,
    gaps: list[str],
    skipped_sources: list[str],
    mix_shortfall: str | None = None,
) -> str:
    """Generate docs/corpus_audit.md content."""
    train = [r for r in records if r.split == "train"]
    holdout = [r for r in records if r.split == "holdout"]
    total_tokens = sum(r.token_count for r in train)
    coverage = metadata_coverage(records)
    heatmap = topic_tradition_heatmap(records)
    mix = compute_mix_buckets(records)
    targets = config.get("mix_buckets", {})

    lines = [
        "# LOGOS Corpus Audit",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Train rows: {len(train)}",
        f"- Holdout rows: {len(holdout)}",
        f"- Train tokens (est.): {total_tokens:,}",
        "",
        "## Metadata coverage (train)",
        "",
    ]
    for field, frac in coverage.items():
        lines.append(f"- `{field}`: {frac:.1%}")
    lines.extend(["", "## Topic × tradition heatmap", "", "| Topic | Tradition | Count |", "|-------|-----------|-------|"])
    for topic in sorted(heatmap):
        for trad, count in sorted(heatmap[topic].items()):
            lines.append(f"| {topic} | {trad} | {count} |")

    lines.extend(["", "## Mix buckets (train tokens)", ""])
    for bucket, target_frac in targets.items():
        actual = mix.get(bucket, 0)
        actual_frac = actual / total_tokens if total_tokens else 0.0
        lines.append(f"- `{bucket}`: {actual:,} tokens ({actual_frac:.1%}) — target {target_frac:.0%}")

    if mix_shortfall:
        lines.extend(["", f"**Mix note:** {mix_shortfall}"])

    if gaps:
        lines.extend(["", "## Gaps flagged", ""])
        for gap in gaps:
            lines.append(f"- {gap}")

    if skipped_sources:
        lines.extend(["", "## Skipped / missing sources", ""])
        for src in skipped_sources:
            lines.append(f"- {src}")

    lines.extend(["", "## Pillar checklist", ""])
    pillars = {
        "Atheist/secular": any(r.tradition in {"atheism", "skepticism", "naturalism", "secular_humanism"} for r in train),
        "Four scriptures": all(
            any(r.source_id.startswith(p) for r in train)
            for p in ("bible_", "quran_", "gita_", "ggs_")
        ),
        "Buddhism (3 schools)": all(
            any(r.school == s for r in train) for s in ("theravada", "zen", "tibetan")
        ),
        "Theist philosophy": any(r.mix_bucket == "theist_philosophy" for r in train),
        "Ethics/logic/psych": any(r.source_id == "logos_fallacies" for r in train),
        "Multi-tradition": any(r.tradition == "jain" for r in train),
        "Debates": any(r.source_type == "debate_transcript" for r in train),
    }
    for pillar, ok in pillars.items():
        status = "OK" if ok else "GAP"
        lines.append(f"- {pillar}: **{status}**")

    return "\n".join(lines) + "\n"


def detect_gaps(records: list[CorpusRecord], registry: dict[str, dict[str, Any]]) -> list[str]:
    """Flag known missing owner files and thin coverage."""
    gaps: list[str] = []
    train = [r for r in records if r.split == "train"]
    pending = [
        ("lewis_mere_christianity", "C.S. Lewis — Mere Christianity"),
        ("chattopadhyaya_charvaka", "Debiprasad Chattopadhyaya — Charvaka/Lokāyata"),
        ("ambedkar_annihilation_caste", "Ambedkar — Annihilation of Caste"),
        ("ambedkar_buddha_dhamma", "Ambedkar — The Buddha and His Dhamma"),
    ]
    present_ids = {r.source_id for r in train}
    for source_id, label in pending:
        if source_id not in present_ids:
            gaps.append(f"Missing (pending owner file): {label}")

    if not any(r.source_id == "mackie_evil_1955" for r in train):
        gaps.append("Mackie problem-of-evil essay not ingested")
    if not any(r.tradition == "jain" for r in train):
        gaps.append("Jain Tattvārtha selections not ingested")
    if not any(r.source_id == "plantinga_gfe" for r in train):
        gaps.append("Plantinga God, Freedom, and Evil not ingested")
    return gaps


def build_corpus(repo_root: Path | None = None) -> list[CorpusRecord]:
    """Run full corpus build pipeline."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    config = load_corpus_config(repo_root / "configs/corpus.yaml")
    registry = load_source_registry(repo_root / "configs/source_registry.yaml")
    paths = config.get("paths", {})
    chunk_cfg = config.get("chunk", {})
    policy = ChunkPolicy(
        target_tokens=int(chunk_cfg.get("target_tokens", 512)),
        min_tokens=int(chunk_cfg.get("min_tokens", 64)),
        max_tokens=int(chunk_cfg.get("max_tokens", 1024)),
    )
    dedup_cfg = config.get("dedup", {})

    records: list[CorpusRecord] = []
    records.extend(ingest_all(repo_root, config_path=repo_root / "configs/corpus.yaml"))
    records.extend(
        ingest_all_transcripts(
            repo_root,
            repo_root / "configs/source_registry.yaml",
            policy,
        )
    )
    records.extend(build_fallacy_records())
    records.extend(build_bias_records())

    topic_registry = repo_root / paths.get("topic_registry", "configs/topic_registry.yaml")
    valid, invalid = filter_valid_records(records, topic_registry_path=topic_registry)
    if invalid:
        sample = invalid[0][1]
        msg = f"{len(invalid)} invalid records; first errors: {sample}"
        raise ValueError(msg)

    kept, _rejected = deduplicate_records(
        valid,
        near_dup_threshold=float(dedup_cfg.get("near_dup_threshold", 0.92)),
        simhash_bits=int(dedup_cfg.get("simhash_bits", 64)),
    )

    ref_texts = load_phase8_reference_texts(repo_root / paths.get("general_shards_dir", "data/general/shards"))
    if ref_texts:
        kept, _phase8_rej = filter_phase8_overlap(
            kept,
            ref_texts,
            threshold=float(dedup_cfg.get("phase8_overlap_threshold", 0.85)),
            simhash_bits=int(dedup_cfg.get("simhash_bits", 64)),
        )

    train = [r for r in kept if r.split == "train"]
    holdout = [r for r in kept if r.split == "holdout"]
    assert_holdout_disjoint(train, holdout)

    train_path = repo_root / paths.get("train_jsonl", "data/logos/train.jsonl")
    holdout_dir = repo_root / paths.get("holdout_dir", "data/logos/holdout")
    fallacies_path = repo_root / paths.get("fallacies_jsonl", "data/logos/fallacies.jsonl")
    biases_path = repo_root / paths.get("biases_jsonl", "data/logos/biases.jsonl")
    audit_path = repo_root / paths.get("audit_report", "docs/corpus_audit.md")

    write_jsonl(train_path, train)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(holdout_dir / "holdout.jsonl", holdout)
    manifest = {
        "generated": datetime.now(UTC).isoformat(),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_ids": [r.id for r in holdout],
    }
    (holdout_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fallacies = [r for r in kept if r.source_id == "logos_fallacies"]
    biases = [r for r in kept if r.source_id == "logos_biases"]
    write_jsonl(fallacies_path, fallacies)
    write_jsonl(biases_path, biases)

    total_tokens = sum(r.token_count for r in train)
    mix_min = int(config.get("mix_min_tokens", 500000))
    mix_shortfall = None
    if total_tokens < mix_min:
        mix_shortfall = (
            f"Corpus has {total_tokens:,} train tokens (< {mix_min:,}); "
            "±5% mix bucket enforcement deferred."
        )
    else:
        tolerance = float(config.get("mix_tolerance", 0.05))
        targets = config.get("mix_buckets", {})
        mix = compute_mix_buckets(kept)
        skewed = []
        for bucket, target_frac in targets.items():
            actual_frac = mix.get(bucket, 0) / total_tokens if total_tokens else 0.0
            if abs(actual_frac - target_frac) > tolerance:
                skewed.append(f"`{bucket}` at {actual_frac:.1%} (target {target_frac:.0%})")
        if skewed:
            mix_shortfall = (
                "Corpus ≥500k tokens but mix buckets outside ±5%: "
                + "; ".join(skewed)
                + ". Scripture-heavy skew expected until licensed canon is rebalanced."
            )

    gaps = detect_gaps(kept, registry)
    audit = generate_audit_report(
        kept,
        config,
        gaps=gaps,
        skipped_sources=[],
        mix_shortfall=mix_shortfall,
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(audit, encoding="utf-8")

    return kept


def main() -> None:
    """CLI entry point for corpus build."""
    records = build_corpus()
    train_count = sum(1 for r in records if r.split == "train")
    print(f"Built {train_count} train records, {len(records) - train_count} holdout")


if __name__ == "__main__":
    main()
