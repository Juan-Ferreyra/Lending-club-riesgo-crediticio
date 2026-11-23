"""Prepare development and sealed external cohorts without inspecting test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.features import RAW_PREDICTORS, IDENTIFIERS, BASE_NUMERIC, BUREAU_NUMERIC


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def prepare(source, sample="data/lending_club_muestra.csv.gz", out="data/processed"):
    out = Path(out)
    if (out/"manifest.json").exists():
        raise FileExistsError("Prepared cohorts already exist; use a separate output directory for a new experiment.")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sample_ids = set(pd.read_csv(sample, usecols=["id"], dtype=str)["id"])
    preview_ids = set(json.loads(Path("experiments/previewed_source_ids.json").read_text(encoding="utf-8"))["ids"])
    excluded_ids = sample_ids | preview_ids
    assert len(sample_ids) == 115000, "The initial sample IDs must be unique."
    seen_sample_ids = set()
    parts = {"development": [], "holdout_2015": [], "holdout_2016": []}
    coverage = []
    preview_coverage = []
    source_rows = 0
    columns = list(dict.fromkeys(IDENTIFIERS + RAW_PREDICTORS))
    for i, chunk in enumerate(pd.read_csv(source, usecols=columns, dtype=str, chunksize=75000)):
        source_rows += len(chunk)
        seen_sample_ids.update(set(chunk.id).intersection(sample_ids))
        chunk["issue_date"] = pd.to_datetime(chunk.issue_d, format="%b-%Y", errors="coerce")
        chunk["issue_year"] = chunk.issue_date.dt.year
        # Coverage is sealed with the holdouts, not printed or used for decisions.
        group = chunk.groupby(["issue_year", "term", "loan_status"], dropna=False).size().reset_index(name="n")
        coverage.append(group)
        preview_rows = chunk[chunk.id.isin(preview_ids) & ~chunk.id.isin(sample_ids)]
        if not preview_rows.empty:
            preview_coverage.append(preview_rows.groupby(["issue_year", "term", "loan_status"], dropna=False).size().reset_index(name="n"))
        chunk = chunk[chunk.loan_status.isin(["Fully Paid", "Charged Off"]) & chunk.id.notna()].copy()
        chunk["target"] = chunk.loan_status.eq("Charged Off").astype("int8")
        for column in BASE_NUMERIC + BUREAU_NUMERIC:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
        parts["development"].append(chunk[chunk.issue_year.between(2007, 2014)])
        for year in [2015, 2016]:
            parts[f"holdout_{year}"].append(chunk[chunk.issue_year.eq(year) & ~chunk.id.isin(excluded_ids)])
        if i % 5 == 0:
            print(f"Read {source_rows:,} source rows; test outcomes remain sealed.", flush=True)
    assert seen_sample_ids == sample_ids, "Original dataset does not contain the full initial sample."
    manifest = {"source_name": Path(source).name, "source_sha256": file_sha256(source),
                "source_rows": source_rows, "initial_sample_sha256": file_sha256(sample),
                "protocol_sha256": file_sha256("experiments/protocol.json"),
                "schema_preview_excluded_ids": sorted(preview_ids),
                "all_sample_ids_found_in_source": True, "partitions": {}}
    ids = {}
    for name, chunks in parts.items():
        frame = pd.concat(chunks, ignore_index=True).sort_values(["issue_date", "id"]).reset_index(drop=True)
        assert frame.id.is_unique
        ids[name] = set(frame.id)
        if name.startswith("holdout"):
            assert not ids[name].intersection(excluded_ids)
        frame.to_parquet(out/f"{name}.parquet", index=False, compression="zstd")
        manifest["partitions"][name] = {"rows": len(frame), "ids_sha256": ids_sha256(frame.id),
            "file_sha256": file_sha256(out/f"{name}.parquet"),
            "original_sample_overlap": len(ids[name].intersection(sample_ids))}
        if name.startswith("holdout"):
            manifest["partitions"][name]["schema_preview_overlap"] = len(ids[name].intersection(preview_ids))
        if name == "development":
            print("Development rows:", len(frame), "positive fraction:", round(frame.target.mean(), 4), flush=True)
    assert not ids["development"].intersection(ids["holdout_2015"] | ids["holdout_2016"])
    assert not ids["holdout_2015"].intersection(ids["holdout_2016"])
    pd.concat(coverage).groupby(["issue_year", "term", "loan_status"], dropna=False)["n"].sum().reset_index().to_parquet(out/"sealed_coverage.parquet", index=False)
    preview_counts = pd.concat(preview_coverage, ignore_index=True) if preview_coverage else pd.DataFrame(columns=["issue_year", "term", "loan_status", "n"])
    preview_counts.to_parquet(out/"sealed_preview_counts.parquet", index=False)
    manifest["preparation_seconds"] = round(time.time()-started, 2)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Cohorts prepared. Test prevalences and outcomes have not been inspected.", flush=True)


def apply_preview_exclusions_to_prepared(out="data/processed"):
    """Upgrade prepared test IDs before freeze, without printing their outcomes."""
    out = Path(out)
    if Path("outputs/selection_lock.json").exists():
        raise RuntimeError("Cannot change external cohorts after model freeze.")
    manifest_path = out/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preview_ids = set(json.loads(Path("experiments/previewed_source_ids.json").read_text(encoding="utf-8"))["ids"])
    if manifest.get("schema_preview_excluded_ids") == sorted(preview_ids):
        print("Schema preview exclusions already applied.")
        return
    counts = []
    for year in [2015, 2016]:
        name = f"holdout_{year}"
        path = out/f"{name}.parquet"
        frame = pd.read_parquet(path)
        observed = frame[frame.id.isin(preview_ids)]
        if not observed.empty:
            counts.append(observed.groupby(["issue_year", "term", "loan_status"]).size().reset_index(name="n"))
        frame = frame[~frame.id.isin(preview_ids)].reset_index(drop=True)
        if len(observed):
            temporary = out/f"{name}.replacement.parquet"
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        partition = manifest["partitions"][name]
        partition.update(rows=len(frame), ids_sha256=ids_sha256(frame.id), file_sha256=file_sha256(path), schema_preview_overlap=0)
        print(f"{year}: removed {len(observed)} schema-preview rows; outcomes not inspected.", flush=True)
    preview_counts = pd.concat(counts, ignore_index=True) if counts else pd.DataFrame(columns=["issue_year", "term", "loan_status", "n"])
    preview_counts.to_parquet(out/"sealed_preview_counts.parquet", index=False)
    manifest["schema_preview_excluded_ids"] = sorted(preview_ids)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    prepare(args.source, out=args.out)
