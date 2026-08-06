#!/usr/bin/env python3
"""Measure clean-to-cued uncertainty shifts on frozen question-disjoint routes.

The input record files are immutable campaign JSONL. Every ``--predictor`` is
evaluated separately; the script never creates a composite predictor. Predictor
scores may live in the campaign rows or in one or more auxiliary JSONL files and
are selected with a dotted field path, for example::

    --predictor msp=msp \
    --predictor mean_token_entropy=confidence_scores.mean_token_entropy \
    --lower-is-more-confident mean_token_entropy

The clean records must contain the already-frozen question-level
``routing_split`` assignment. The analysis validates and consumes it; it never
reassigns a question. Test statistics use only exact clean/cued pairs for which
the selected predictor is available on both sides.

Every supplied cued row must belong to the frozen test split. The analyzer
requires the complete declared authority and bandwagon dose grid and refuses
partial condition files. Clean ties remain in clean calibration to preserve the
prior threshold-fitting estimand. In test, only rows whose cue reference is the
model's binary clean verdict enter the primary target-bias cohort. The
``--include-clean-ties`` option instead requests the explicitly labeled
fallback-reference robustness sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from biases.analysis.records import ConditionRecord, record_from_mapping
from biases.analysis.uncertainty_shift import (
    PredictorSpec,
    controlled_uncertainty_shift_report,
)
from biases.social_cue_prompts import AUTHORITY_DOSES, BANDWAGON_DOSES
from biases.dataset_splits import routing_assignment_sha256, routing_manifest
from biases.position_bias import load_position_pairs_with_eligibility


ROUTING_SCHEMA_VERSION = 2
ROUTING_ARTIFACT_TYPE = "frozen_question_disjoint_routing_package"
ROUTING_SEED = 42
CALIBRATION_FRACTION = 0.5
ROUTING_OUTPUT_NAMES = ("full", "calibration", "test")
ROUTING_OUTPUT_FILENAMES = {
    "full": "routed_full.csv",
    "calibration": "routed_calibration.csv",
    "test": "routed_test.csv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            record_id = str(row.get("record_id") or "")
            if not record_id:
                raise ValueError(f"{path}:{line_number}: record_id is required")
            if record_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate record_id {record_id!r}")
            seen.add(record_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without treating integers and floats as equal."""

    return json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_frozen_routing_package(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any], Path]:
    """Load and independently verify the raw question-routing universe."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("routing manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("routing manifest must contain one JSON object")
    if (
        manifest.get("schema_version") != ROUTING_SCHEMA_VERSION
        or manifest.get("artifact_type") != ROUTING_ARTIFACT_TYPE
    ):
        raise ValueError("routing manifest is not a supported schema-2 package")
    if manifest.get("routing_unit") != "question":
        raise ValueError("routing manifest must freeze question-level routing")
    if (
        type(manifest.get("seed")) is not int
        or manifest.get("seed") != ROUTING_SEED
    ):
        raise ValueError(f"routing manifest must freeze seed={ROUTING_SEED}")
    if (
        type(manifest.get("calibration_fraction")) is not float
        or manifest.get("calibration_fraction") != CALIBRATION_FRACTION
    ):
        raise ValueError(
            "routing manifest must freeze calibration_fraction="
            f"{CALIBRATION_FRACTION}"
        )
    outputs = manifest.get("outputs")
    expected_output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != set(ROUTING_OUTPUT_NAMES)
        or any(not isinstance(outputs.get(name), Mapping) for name in ROUTING_OUTPUT_NAMES)
    ):
        raise ValueError("routing manifest must contain exactly three output records")
    if (
        not isinstance(expected_output_hashes, Mapping)
        or set(expected_output_hashes) != set(ROUTING_OUTPUT_NAMES)
    ):
        raise ValueError("routing manifest must hash exactly three routing outputs")
    output_paths: dict[str, Path] = {}
    output_frames: dict[str, pd.DataFrame] = {}
    for name in ROUTING_OUTPUT_NAMES:
        record = outputs[name]
        assert isinstance(record, Mapping)
        raw_path_value = record.get("path")
        if not isinstance(raw_path_value, str):
            raise ValueError(
                f"routing manifest outputs.{name}.path must be one filename"
            )
        raw_path = Path(raw_path_value)
        if (
            raw_path.is_absolute()
            or len(raw_path.parts) != 1
            or raw_path.name != ROUTING_OUTPUT_FILENAMES[name]
        ):
            raise ValueError(
                f"routing manifest outputs.{name}.path must be one filename"
            )
        path = (manifest_path.parent / raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = expected_output_hashes[name]
        if (
            not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or expected_hash != file_sha256(path)
        ):
            raise ValueError(
                f"routed {name} CSV hash does not match the routing manifest"
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if (
            type(record.get("rows")) is not int
            or record.get("rows") != len(frame)
        ):
            raise ValueError(
                f"routing manifest outputs.{name}.rows does not match the CSV"
            )
        output_paths[name] = path
        output_frames[name] = frame

    full_path = output_paths["full"]
    frame = output_frames["full"]
    required_columns = {"question_id", "routing_split"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("routed full CSV lacks question_id/routing_split")
    for name in ("calibration", "test"):
        if list(output_frames[name].columns) != list(frame.columns):
            raise ValueError(f"routed {name} CSV columns differ from full CSV")
        expected_frame = frame.loc[
            frame["routing_split"].astype(str).str.strip().str.lower() == name
        ].reset_index(drop=True)
        if not output_frames[name].reset_index(drop=True).equals(expected_frame):
            raise ValueError(f"routed {name} CSV is not the exact full-CSV partition")

    raw_assignments: dict[str, str] = {}
    for question_id, routing_split in frame[
        ["question_id", "routing_split"]
    ].itertuples(index=False, name=None):
        question = str(question_id).strip()
        split = str(routing_split).strip().lower()
        if not question or split not in {"calibration", "test"}:
            raise ValueError("routed full CSV contains invalid question routing")
        previous = raw_assignments.setdefault(question, split)
        if previous != split:
            raise ValueError(f"question {question!r} spans both routing splits")
    observed_routing = routing_manifest(
        frame,
        routing_unit="question",
        seed=ROUTING_SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    for field in (
        "routing_unit",
        "seed",
        "calibration_fraction",
        "row_counts",
        "question_counts",
        "routing_assignment_sha256",
    ):
        if not _exact_json_equal(observed_routing[field], manifest.get(field)):
            raise ValueError(f"routing manifest {field} does not match full CSV")
    raw_sha256 = observed_routing["routing_assignment_sha256"]
    if raw_sha256 != manifest.get("routing_assignment_sha256"):
        raise ValueError("raw question-routing SHA-256 does not match the manifest")

    pairs, eligibility = load_position_pairs_with_eligibility(full_path)
    eligible_assignments: dict[str, str] = {}
    for pair in pairs:
        metadata = pair.original.metadata
        question = str(
            metadata.get("question_cluster_id") or pair.original.question_id
        ).strip()
        split = str(metadata.get("routing_split") or "").strip().lower()
        if not question or split not in {"calibration", "test"}:
            raise ValueError("canonical eligible pair has invalid question routing")
        previous = eligible_assignments.setdefault(question, split)
        if previous != split:
            raise ValueError(
                f"eligible question {question!r} spans both routing splits"
            )
    if not eligible_assignments:
        raise ValueError("routing package contains no eligible question")
    eligible_frame = pd.DataFrame(
        [
            {"question_id": question, "routing_split": eligible_assignments[question]}
            for question in sorted(eligible_assignments)
        ]
    )
    eligible_sha256 = routing_assignment_sha256(eligible_frame)
    eligible_values = list(eligible_assignments.values())
    expected_counts = {
        "raw_rows": dict(observed_routing["row_counts"]),
        "raw_questions": dict(observed_routing["question_counts"]),
        "eligible_pairs": {
            "total": eligibility.eligible_pair_count,
            "calibration": int(
                eligibility.routing_counts["eligible_pairs"].get("calibration", 0)
            ),
            "test": int(
                eligibility.routing_counts["eligible_pairs"].get("test", 0)
            ),
        },
        "eligible_questions": {
            "total": len(eligible_assignments),
            "calibration": sum(value == "calibration" for value in eligible_values),
            "test": sum(value == "test" for value in eligible_values),
            "overlap": 0,
        },
        "skipped_rows": {
            "total": eligibility.skipped_row_count,
            "calibration": int(
                eligibility.routing_counts["skipped_rows"].get("calibration", 0)
            ),
            "test": int(
                eligibility.routing_counts["skipped_rows"].get("test", 0)
            ),
        },
    }
    if not _exact_json_equal(manifest.get("counts"), expected_counts):
        raise ValueError("routing manifest raw/eligible/skipped counts do not match")
    if not _exact_json_equal(manifest.get("eligibility"), eligibility.to_dict()):
        raise ValueError("routing manifest eligibility hash/audit does not match")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "full_csv_path": str(full_path),
        "full_csv_sha256": file_sha256(full_path),
        "raw_routing_assignment_sha256": raw_sha256,
        "eligible_routing_assignment_sha256": eligible_sha256,
        "raw_question_count": len(raw_assignments),
        "eligible_question_count": len(eligible_assignments),
        "eligibility_sha256": eligibility.eligibility_sha256,
    }
    return raw_assignments, eligible_assignments, provenance, full_path


def parse_predictor_declarations(
    declarations: list[str],
    lower_is_more_confident: set[str],
) -> tuple[tuple[PredictorSpec, ...], dict[str, str]]:
    predictors: list[PredictorSpec] = []
    fields: dict[str, str] = {}
    for declaration in declarations:
        if "=" not in declaration:
            raise ValueError("predictors must use NAME=DOTTED_FIELD syntax")
        name, field = (part.strip() for part in declaration.split("=", 1))
        if not name or not field:
            raise ValueError("predictor name and dotted field must not be empty")
        if name in fields:
            raise ValueError(f"duplicate predictor {name!r}")
        fields[name] = field
        predictors.append(
            PredictorSpec(
                name=name,
                higher_is_more_confident=name not in lower_is_more_confident,
            )
        )
    unknown = lower_is_more_confident - set(fields)
    if unknown:
        raise ValueError(
            "--lower-is-more-confident names undeclared predictors: "
            + ", ".join(sorted(unknown))
        )
    return tuple(predictors), fields


def nested_value(row: Mapping[str, Any], dotted_field: str) -> Any:
    value: Any = row
    for component in dotted_field.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _numeric_value(value: Any, *, predictor: str, record_id: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{predictor} score for {record_id!r} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{predictor} score for {record_id!r} is not numeric"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{predictor} score for {record_id!r} is not finite")
    return result


def build_score_table(
    source_rows: list[list[dict[str, Any]]],
    predictor_fields: Mapping[str, str],
) -> dict[str, dict[str, float | None]]:
    """Join selected fields by record ID and reject conflicting duplicates."""

    sources_by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rows in source_rows:
        for row in rows:
            sources_by_record[str(row["record_id"])].append(row)
    table: dict[str, dict[str, float | None]] = {}
    for record_id, sources in sources_by_record.items():
        table[record_id] = {}
        for predictor, field in predictor_fields.items():
            values = [
                nested_value(source, field)
                for source in sources
                if nested_value(source, field) is not None
            ]
            numeric = [
                _numeric_value(value, predictor=predictor, record_id=record_id)
                for value in values
            ]
            if numeric and any(
                not math.isclose(value, numeric[0], rel_tol=0.0, abs_tol=1e-15)
                for value in numeric[1:]
            ):
                raise ValueError(
                    f"conflicting {predictor} scores for record {record_id!r}"
                )
            table[record_id][predictor] = numeric[0] if numeric else None
    return table


def parse_records(
    rows: list[dict[str, Any]],
    *,
    expected_family: str | None,
) -> tuple[ConditionRecord, ...]:
    records: list[ConditionRecord] = []
    for row in rows:
        record = record_from_mapping(row)
        if not record.record_id:
            raise ValueError("parsed record has an empty record_id")
        if expected_family is not None and record.family != expected_family:
            raise ValueError(
                f"record {record.record_id!r} has family {record.family!r}; "
                f"expected {expected_family!r}"
            )
        records.append(record)
    return tuple(records)


def validate_output_path(output: Path, inputs: list[Path]) -> None:
    resolved_output = output.resolve()
    if any(resolved_output == path.resolve() for path in inputs):
        raise ValueError("output path must differ from every input path")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")


def validate_lowercase_sha256(value: str, *, name: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be exactly 64 lowercase hex digits")
    return value


def validate_model_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(
            "expected model revision must be a lowercase 40-hex commit"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-records", type=Path, required=True)
    parser.add_argument("--cued-records", type=Path, required=True)
    parser.add_argument(
        "--score-records",
        type=Path,
        action="append",
        default=[],
        help="Auxiliary record-keyed score JSONL; repeat as needed.",
    )
    parser.add_argument(
        "--predictor",
        action="append",
        required=True,
        help="Separate predictor declaration NAME=DOTTED_FIELD.",
    )
    parser.add_argument(
        "--lower-is-more-confident",
        action="append",
        default=[],
        metavar="NAME",
    )
    parser.add_argument(
        "--target-risk",
        action="append",
        type=float,
        dest="target_risks",
    )
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument(
        "--expected-raw-routing-assignment-sha256",
        help="Optional SHA-256 of all raw routed source questions.",
    )
    parser.add_argument(
        "--expected-eligible-routing-assignment-sha256",
        help="Optional SHA-256 of the exact canonically eligible question routes.",
    )
    parser.add_argument(
        "--routing-manifest",
        type=Path,
        help=(
            "Schema-2 frozen routing package. Required for release analyses "
            "whose raw routing universe includes ineligible source rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--include-clean-ties",
        action="store_true",
        help=(
            "Run the labeled robustness sensitivity that includes clean-tie "
            "rows aimed by human-label or deterministic fallback references. "
            "These rows are never called primary target-bias. Clean calibration "
            "ties remain included in either mode."
        ),
    )
    parser.add_argument(
        "--authority-dose",
        action="append",
        type=int,
        choices=AUTHORITY_DOSES,
        help="Declared authority dose; repeat to override the full project grid.",
    )
    parser.add_argument(
        "--bandwagon-dose",
        action="append",
        type=int,
        choices=BANDWAGON_DOSES,
        help="Declared bandwagon dose; repeat to override the full project grid.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [args.clean_records, args.cued_records, *args.score_records]
    frozen_raw_assignments: dict[str, str] | None = None
    frozen_eligible_assignments: dict[str, str] | None = None
    routing_provenance: dict[str, Any] | None = None
    if args.routing_manifest is not None:
        if not args.routing_manifest.is_file():
            raise FileNotFoundError(args.routing_manifest)
        (
            frozen_raw_assignments,
            frozen_eligible_assignments,
            routing_provenance,
            routed_full_path,
        ) = load_frozen_routing_package(args.routing_manifest)
        input_paths.extend((args.routing_manifest, routed_full_path))
    if any(not path.is_file() for path in input_paths):
        missing = [str(path) for path in input_paths if not path.is_file()]
        raise FileNotFoundError("missing input files: " + ", ".join(missing))
    if args.output is not None:
        validate_output_path(args.output, input_paths)
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must be in (0, 1)")
    expected_model_name = args.expected_model_name.strip()
    if not expected_model_name:
        raise ValueError("expected model name must not be blank")
    expected_model_revision = validate_model_revision(
        args.expected_model_revision
    )
    expected_raw_sha = args.expected_raw_routing_assignment_sha256
    if expected_raw_sha is not None:
        expected_raw_sha = validate_lowercase_sha256(
            expected_raw_sha,
            name="expected raw routing assignment SHA-256",
        )
    expected_eligible_sha = args.expected_eligible_routing_assignment_sha256
    if expected_eligible_sha is not None:
        expected_eligible_sha = validate_lowercase_sha256(
            expected_eligible_sha,
            name="expected eligible routing assignment SHA-256",
        )
    if routing_provenance is not None:
        package_raw_sha = routing_provenance["raw_routing_assignment_sha256"]
        package_eligible_sha = routing_provenance[
            "eligible_routing_assignment_sha256"
        ]
        if expected_raw_sha is not None and expected_raw_sha != package_raw_sha:
            raise ValueError(
                "expected raw routing assignment SHA-256 does not match the package"
            )
        if (
            expected_eligible_sha is not None
            and expected_eligible_sha != package_eligible_sha
        ):
            raise ValueError(
                "expected eligible routing assignment SHA-256 does not match "
                "the package"
            )
        expected_raw_sha = package_raw_sha
        expected_eligible_sha = package_eligible_sha

    predictors, predictor_fields = parse_predictor_declarations(
        args.predictor,
        set(args.lower_is_more_confident),
    )
    clean_rows = read_jsonl(args.clean_records)
    cued_rows = read_jsonl(args.cued_records)
    overlap = {str(row["record_id"]) for row in clean_rows} & {
        str(row["record_id"]) for row in cued_rows
    }
    if overlap:
        raise ValueError(f"clean and cued record IDs overlap: {sorted(overlap)[:10]!r}")
    score_rows = [read_jsonl(path) for path in args.score_records]
    scores = build_score_table(
        [clean_rows, cued_rows, *score_rows],
        predictor_fields,
    )
    clean_records = parse_records(clean_rows, expected_family="clean")
    cued_records = parse_records(cued_rows, expected_family=None)
    if any(record.family == "clean" for record in cued_records):
        raise ValueError("cued record input contains clean-family rows")

    report = controlled_uncertainty_shift_report(
        clean_records,
        cued_records,
        scores,
        predictors,
        target_risks=tuple(args.target_risks or (0.10, 0.20)),
        expected_raw_assignment_sha256=expected_raw_sha,
        expected_eligible_assignment_sha256=expected_eligible_sha,
        frozen_raw_question_assignments=frozen_raw_assignments,
        frozen_eligible_question_assignments=frozen_eligible_assignments,
        expected_model_name=expected_model_name,
        expected_model_revision=expected_model_revision,
        seed=args.seed,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        exclude_clean_ties=not args.include_clean_ties,
        authority_doses=tuple(args.authority_dose or AUTHORITY_DOSES),
        bandwagon_doses=tuple(args.bandwagon_dose or BANDWAGON_DOSES),
    )
    report["provenance"] = {
        "input_sha256": {
            str(path): file_sha256(path) for path in input_paths
        },
        "predictor_fields": predictor_fields,
        "routing_package": routing_provenance,
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
