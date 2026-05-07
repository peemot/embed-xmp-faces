#!/usr/bin/env python3.10
"""Embed face metadata from .xmp sidecars into image files using ExifTool."""

import argparse
import copy
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

LOGGER_NAME = "FaceEmbedder"
EXIFTOOL_EXECUTABLE = "exiftool"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REGION_INFO_KEEP, REGION_INFO_CLEAR, REGION_INFO_WRITE = "keep", "clear", "write"
OVERLAP_THRESHOLD, CONTAINMENT_THRESHOLD = 0.70, 0.85
UTF8_CHARSET_ARGS = ("-charset", "utf8", "-charset", "filename=utf8")

MODE_FILL, MODE_MERGE, MODE_REPLACE = "fill", "merge", "replace"
VALID_MODES = (MODE_FILL, MODE_MERGE, MODE_REPLACE)

DECISION_SKIP, DECISION_UPDATE, DECISION_UNCHANGED = "skip", "update", "unchanged"
WRITE_RESULT_NOT_APPLICABLE, WRITE_RESULT_DRY_RUN = "not_applicable", "dry_run"
WRITE_RESULT_UPDATED, WRITE_RESULT_UNCHANGED, WRITE_RESULT_FAILED = "updated", "unchanged", "failed"

REASON_MISSING_METADATA, REASON_XMP_EMPTY = "missing_metadata", "xmp_empty"
REASON_IMAGE_HAS_EXISTING_METADATA, REASON_NO_CHANGE = "image_has_existing_metadata", "no_change"
REASON_OVERLAP_DIFFERENT_REAL_NAMES = "overlap_different_real_names"
MetadataMap = dict[str, dict[str, Any]]
ActionsMap = CountsMap = dict[str, int]

@dataclass
class WriteTask:
    image_path: str
    subjects: list[str]
    region_info_action: str
    region_info: dict[str, Any] | None = None

@dataclass
class EvaluatedFile:
    image_path: str
    xmp_path: str
    reasons: list[str]
    counts: CountsMap
    actions: ActionsMap
    decision: str = DECISION_SKIP
    write_result: str = WRITE_RESULT_NOT_APPLICABLE
    error: str | None = None

ZERO_ACTIONS: ActionsMap = {
    "xmp_regions_added": 0, "xmp_regions_from_xmp": 0, "image_regions_kept": 0,
    "image_regions_removed": 0, "regions_collapsed": 0,
    "conflicts_kept_both": 0, "unnamed_regions_written": 0,
}
ZERO_COUNTS: CountsMap = {
    "image_subjects": 0,
    "image_regions": 0,
    "xmp_subjects": 0,
    "xmp_regions": 0,
    "final_subjects": 0, "final_regions": 0,
}

def setup_logger(debug_mode: bool, timestamp: str) -> tuple[logging.Logger, str | None]:
    """Set up console logging and the optional debug file."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    debug_log_filename: str | None = None
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    if debug_mode:
        debug_log_filename = os.path.join(log_dir, f"face_embed_debug_{timestamp}.txt")
        debug_handler = logging.FileHandler(debug_log_filename, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(debug_handler)

    return logger, debug_log_filename

def find_image_xmp_pairs(root_dir: str) -> list[tuple[str, str]]:
    """Recursively find images that have an exact matching .xmp sidecar."""
    pairs: list[tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if not filenames:
            continue
        lower_to_actual = {filename.lower(): filename for filename in filenames}
        abs_dirpath = os.path.abspath(dirpath)
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
            expected_xmp_lower = f"{filename}.xmp".lower()
            if expected_xmp_lower not in lower_to_actual:
                continue
            actual_xmp_filename = lower_to_actual[expected_xmp_lower]
            image_path = os.path.join(abs_dirpath, filename)
            xmp_path = os.path.join(abs_dirpath, actual_xmp_filename)
            pairs.append((image_path, xmp_path))
    return pairs

def normalize_list(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, (list, tuple, set)):
        return list(data)
    return [data]

def subject_key(subject: str) -> str:
    return subject.casefold()

def normalize_person_name(value: Any) -> str | None:
    """Normalize person-like values, treating blank/Unknown as unnamed."""
    if value is None:
        return None
    name = str(value).strip()
    if not name or name.casefold() == "unknown":
        return None
    return name

def normalize_subjects(data: Any) -> list[str]:
    """Normalize Subject values, dropping blanks and literal Unknown entries."""
    subjects: list[str] = []
    seen: set[str] = set()
    for item in normalize_list(data):
        subject = normalize_person_name(item)
        if subject is None:
            continue
        key = subject_key(subject)
        if key in seen:
            continue
        seen.add(key)
        subjects.append(subject)
    return subjects

def region_is_face_like(region: dict[str, Any]) -> bool:
    """Treat blank type and explicit Face type as face-like for name handling."""
    region_type = str(region.get("Type", "")).strip().casefold()
    return not region_type or region_type == "face"

def region_real_name(region: dict[str, Any]) -> str | None:
    """Return the normalized real name for a face-like region, if any."""
    if not region_is_face_like(region):
        return None
    return normalize_person_name(region.get("Name"))

def normalize_region(region: dict[str, Any]) -> dict[str, Any]:
    """Normalize one region dictionary without changing non-face naming behavior."""
    normalized_region = copy.deepcopy(region)
    if region_is_face_like(normalized_region):
        normalized_name = normalize_person_name(normalized_region.get("Name"))
        if normalized_name is None:
            normalized_region.pop("Name", None)
        else:
            normalized_region["Name"] = normalized_name
    return normalized_region

def normalize_region_info(region_info: Any, source_path: str) -> dict[str, Any]:
    """Normalize RegionInfo into a predictable internal dictionary shape."""
    if region_info is None:
        return {}
    if not isinstance(region_info, dict):
        raise ValueError(f"Invalid RegionInfo in {source_path}: expected an object.")
    normalized: dict[str, Any] = {}
    applied_dims = region_info.get("AppliedToDimensions")
    if applied_dims is not None:
        if not isinstance(applied_dims, dict):
            raise ValueError(
                f"Invalid RegionInfo.AppliedToDimensions in {source_path}: expected an object."
            )
        if applied_dims:
            normalized["AppliedToDimensions"] = copy.deepcopy(applied_dims)
    raw_regions = region_info.get("RegionList")
    regions: list[dict[str, Any]] = []
    for index, region in enumerate(normalize_list(raw_regions)):
        if not isinstance(region, dict):
            raise ValueError(
                f"Invalid RegionInfo.RegionList[{index}] in {source_path}: expected an object."
            )
        regions.append(copy.deepcopy(region))
    if regions:
        normalized["RegionList"] = regions
    return normalized

def safe_normalize_region_info(
    region_info: Any,
    source_path: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Normalize RegionInfo, degrading malformed data to a warning and fallback."""
    try:
        return normalize_region_info(region_info, source_path)
    except ValueError as exc:
        logger.warning("Ignoring malformed RegionInfo in %s: %s", source_path, exc)
        return {}

def prepare_pair_metadata(
    image_meta: dict[str, Any],
    xmp_meta: dict[str, Any],
    image_path: str,
    xmp_path: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract normalized metadata plus raw compare values for the image."""
    image_region_info = safe_normalize_region_info(image_meta.get("RegionInfo"), image_path, logger)
    xmp_region_info = safe_normalize_region_info(xmp_meta.get("RegionInfo"), xmp_path, logger)
    image_regions = [normalize_region(region) for region in image_region_info.get("RegionList", [])]
    xmp_regions = deduplicate_regions(
        [normalize_region(region) for region in xmp_region_info.get("RegionList", [])]
    )
    return (
        {
            "subjects": normalize_subjects(image_meta.get("Subject")),
            "compare_subjects": [
                str(item) for item in normalize_list(image_meta.get("Subject")) if item is not None
            ],
            "regions": image_regions,
            "compare_region_info": image_region_info,
            "dimensions": image_region_info.get("AppliedToDimensions"),
        },
        {
            "subjects": normalize_subjects(xmp_meta.get("Subject")),
            "regions": xmp_regions,
            "dimensions": xmp_region_info.get("AppliedToDimensions"),
        },
    )

def encode_exiftool_struct(value: Any) -> str:
    """Encode Python dict/list structures in ExifTool's struct syntax."""
    if isinstance(value, dict):
        parts = [
            f"{key}={encode_exiftool_struct(subvalue)}"
            for key, subvalue in value.items()
        ]
        return "{" + ",".join(parts) + "}"

    if isinstance(value, list):
        parts = [encode_exiftool_struct(item) for item in value]
        return "[" + ",".join(parts) + "]"

    serialized = str(value)
    serialized = (
        serialized.replace("|", "||")
        .replace(",", "|,")
        .replace("}", "|}")
        .replace("]", "|]")
    )
    if serialized and serialized[0] in "{[ \t\r\n":
        serialized = "|" + serialized
    return serialized

def build_exiftool_command(
    argfile_path: str,
    base_args: Sequence[str] | None = None,
    common_args: Sequence[str] | None = None,
) -> list[str]:
    """Build an ExifTool command for an argfile."""
    command = [EXIFTOOL_EXECUTABLE]
    if base_args:
        command.extend(base_args)
    command.extend(["-@", argfile_path])
    if common_args:
        command.append("-common_args")
        command.extend(common_args)
    return command

def run_exiftool(
    command: Sequence[str],
    logger: logging.Logger,
    phase: str,
) -> subprocess.CompletedProcess | None:
    """Execute ExifTool with clearer dependency and OS-level error reporting."""
    logger.debug("Executing %s command: %s", phase, " ".join(command))

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        logger.error(
            "ExifTool executable '%s' was not found. Install ExifTool and ensure it is on PATH.",
            EXIFTOOL_EXECUTABLE,
        )
        return None
    except OSError as exc:
        logger.error("Failed to execute ExifTool during %s: %s", phase, exc)
        return None

    if result.stdout:
        logger.debug("ExifTool %s STDOUT:\n%s", phase, result.stdout)
    if result.stderr:
        logger.debug("ExifTool %s STDERR:\n%s", phase, result.stderr)

    return result

def build_metadata_map(
    raw_metadata: Any,
    expected_files: set[str],
    logger: logging.Logger,
) -> MetadataMap | None:
    """Build a SourceFile -> metadata map from Pass 1 output."""
    if not isinstance(raw_metadata, list):
        logger.error("Pass 1 did not return a JSON array as expected.")
        return None

    source_paths: list[str] = []
    metadata_map: MetadataMap = {}

    for index, item in enumerate(raw_metadata):
        if not isinstance(item, dict):
            logger.error("Pass 1 returned a non-object JSON item at index %d.", index)
            return None

        source_file = item.get("SourceFile")
        if not isinstance(source_file, str) or not source_file:
            logger.error("Pass 1 returned an item without a valid SourceFile at index %d.", index)
            return None

        absolute_source = os.path.abspath(source_file)
        source_paths.append(absolute_source)
        metadata_map[absolute_source] = item

    duplicate_paths = [path for path, count in Counter(source_paths).items() if count > 1]
    if duplicate_paths:
        logger.error(
            "Pass 1 returned duplicate metadata entries: %s",
            "; ".join(sorted(duplicate_paths)),
        )
        return None

    actual_files = set(source_paths)
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)

    if missing_files:
        logger.warning(
            "Pass 1 is missing metadata for %d expected file(s); affected pair(s) will be skipped.",
            len(missing_files),
        )
        for path in missing_files:
            logger.warning("Missing metadata: %s", path)

    if unexpected_files:
        logger.warning(
            "Pass 1 returned metadata for %d unexpected file(s); they will be ignored.",
            len(unexpected_files),
        )
        for path in unexpected_files:
            logger.warning("Unexpected metadata entry: %s", path)

    return metadata_map

def read_pass(
    pairs: Sequence[tuple[str, str]],
    logger: logging.Logger,
) -> MetadataMap | None:
    """Read metadata for all image/XMP pairs in one ExifTool pass."""
    expected_files = {path for pair in pairs for path in pair}

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
    ) as handle:
        for image_path, xmp_path in pairs:
            handle.write(f"{image_path}\n{xmp_path}\n")
        read_args_file = handle.name

    try:
        command = build_exiftool_command(
            read_args_file,
            base_args=["-q", *UTF8_CHARSET_ARGS, "-j", "-struct", "-Subject", "-RegionInfo"],
        )
        result = run_exiftool(command, logger, "Pass 1")
        if result is None:
            return None

        stdout_text = result.stdout.strip()
        stderr_text = result.stderr.strip()

        if result.returncode != 0:
            if not stdout_text:
                logger.error("ExifTool read failed with exit status %s.", result.returncode)
                if stderr_text:
                    logger.error("ExifTool read error output:\n%s", stderr_text)
                return None

            logger.warning(
                "ExifTool read exited with status %s; continuing with returned JSON and "
                "skipping any missing pair(s).",
                result.returncode,
            )
            if stderr_text:
                logger.warning("ExifTool read diagnostics:\n%s", stderr_text)
        elif stderr_text:
            logger.warning("ExifTool read reported warnings:\n%s", stderr_text)

        if not stdout_text:
            logger.error("ExifTool read returned no JSON output.")
            return None

        try:
            raw_metadata = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Pass 1 JSON output: %s", exc)
            return None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "--- RAW READ METADATA ---\n%s\n-------------------------",
                json.dumps(raw_metadata, indent=2, ensure_ascii=False),
            )

        return build_metadata_map(raw_metadata, expected_files, logger)
    finally:
        if os.path.exists(read_args_file):
            os.remove(read_args_file)

def calculate_overlap_metrics(
    area1: dict[str, Any],
    area2: dict[str, Any],
) -> tuple[float, float]:
    """Return (IoU, overlap fraction of the smaller box)."""
    try:
        x1, y1 = float(area1.get("X", 0)), float(area1.get("Y", 0))
        w1, h1 = float(area1.get("W", 0)), float(area1.get("H", 0))

        x2, y2 = float(area2.get("X", 0)), float(area2.get("Y", 0))
        w2, h2 = float(area2.get("W", 0)), float(area2.get("H", 0))
    except (TypeError, ValueError):
        return 0.0, 0.0

    if min(w1, h1, w2, h2) <= 0:
        return 0.0, 0.0

    l1, r1 = x1 - (w1 / 2), x1 + (w1 / 2)
    t1, b1 = y1 - (h1 / 2), y1 + (h1 / 2)

    l2, r2 = x2 - (w2 / 2), x2 + (w2 / 2)
    t2, b2 = y2 - (h2 / 2), y2 + (h2 / 2)

    inter_l = max(l1, l2)
    inter_r = min(r1, r2)
    inter_t = max(t1, t2)
    inter_b = min(b1, b2)

    if inter_l >= inter_r or inter_t >= inter_b:
        return 0.0, 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    area1_size = w1 * h1
    area2_size = w2 * h2
    union_area = area1_size + area2_size - inter_area

    iou = (inter_area / union_area) if union_area > 0 else 0.0
    smaller_overlap = inter_area / min(area1_size, area2_size)
    return iou, smaller_overlap

def is_significant_overlap(
    reg1: dict[str, Any],
    reg2: dict[str, Any],
    threshold: float = OVERLAP_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> bool:
    """Check if two regions likely describe the same face/object."""
    type1 = str(reg1.get("Type", "")).strip().casefold()
    type2 = str(reg2.get("Type", "")).strip().casefold()

    if type1 and type2 and type1 != type2:
        return False

    area1 = reg1.get("Area") if isinstance(reg1.get("Area"), dict) else {}
    area2 = reg2.get("Area") if isinstance(reg2.get("Area"), dict) else {}

    iou, smaller_overlap = calculate_overlap_metrics(area1, area2)
    return iou >= threshold or smaller_overlap >= containment_threshold

def deduplicate_regions(
    regions: list[dict[str, Any]],
    threshold: float = OVERLAP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Remove overlapping duplicate regions, keeping the most recently added one."""
    unique_regions: list[dict[str, Any]] = []

    for candidate in reversed(regions):
        has_overlap = False
        for accepted in unique_regions:
            if is_significant_overlap(candidate, accepted, threshold):
                has_overlap = True
                break
        if not has_overlap:
            unique_regions.append(copy.deepcopy(candidate))

    return list(reversed(unique_regions))

def build_final_subjects(
    mode: str,
    image_subjects: Sequence[str],
    xmp_subjects: Sequence[str],
    final_regions: Sequence[dict[str, Any]],
) -> list[str]:
    """Construct the final Subject list from mode rules and final resolved regions."""
    final_subjects: list[str] = []
    seen_subjects: set[str] = set()

    def add_subjects(subjects: Sequence[str]) -> None:
        for subject in subjects:
            key = subject_key(subject)
            if key in seen_subjects:
                continue
            seen_subjects.add(key)
            final_subjects.append(subject)

    if mode == MODE_MERGE:
        add_subjects(image_subjects)
    add_subjects(xmp_subjects)
    add_subjects(
        [
            real_name
            for real_name in (region_real_name(region) for region in final_regions)
            if real_name is not None
        ]
    )
    return final_subjects

def merge_regions(
    xmp_regions: Sequence[dict[str, Any]],
    image_regions: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ActionsMap, list[str]]:
    """Merge image regions into XMP-first regions using the requested face rules."""
    final_regions = [copy.deepcopy(region) for region in xmp_regions]
    base_xmp_count = len(final_regions)
    actions = ZERO_ACTIONS.copy()
    actions["xmp_regions_added"] = actions["xmp_regions_from_xmp"] = len(xmp_regions)
    reasons: list[str] = []

    for image_region in image_regions:
        overlap_index: int | None = None
        for index in range(base_xmp_count):
            if is_significant_overlap(final_regions[index], image_region):
                overlap_index = index
                break

        if overlap_index is None:
            final_regions.append(copy.deepcopy(image_region))
            actions["image_regions_kept"] += 1
            continue

        xmp_region = final_regions[overlap_index]
        if not (region_is_face_like(xmp_region) and region_is_face_like(image_region)):
            actions["image_regions_removed"] += 1
            continue

        xmp_name = region_real_name(xmp_region)
        image_name = region_real_name(image_region)
        if xmp_name is not None and image_name is not None:
            if subject_key(xmp_name) == subject_key(image_name):
                actions["image_regions_removed"] += 1
                actions["regions_collapsed"] += 1
                continue

            final_regions.append(copy.deepcopy(image_region))
            actions["image_regions_kept"] += 1
            actions["conflicts_kept_both"] += 1
            if REASON_OVERLAP_DIFFERENT_REAL_NAMES not in reasons:
                reasons.append(REASON_OVERLAP_DIFFERENT_REAL_NAMES)
            continue

        if xmp_name is not None and image_name is None:
            actions["image_regions_removed"] += 1
            actions["regions_collapsed"] += 1
            continue

        kept_region = copy.deepcopy(xmp_region)
        if image_name is None:
            kept_region.pop("Name", None)
        else:
            kept_region["Name"] = image_name
        final_regions[overlap_index] = kept_region
        actions["image_regions_removed"] += 1
        actions["regions_collapsed"] += 1

    return final_regions, actions, reasons

def prepare_write_tasks(
    pairs: Sequence[tuple[str, str]],
    metadata_map: MetadataMap,
    mode: str,
    logger: logging.Logger,
) -> tuple[list[WriteTask], list[EvaluatedFile]]:
    """Evaluate metadata and prepare final write operations plus audit entries."""
    write_tasks: list[WriteTask] = []
    evaluations: list[EvaluatedFile] = []

    for image_path, xmp_path in pairs:
        evaluation = EvaluatedFile(
            image_path=image_path,
            xmp_path=xmp_path,
            reasons=[],
            counts=ZERO_COUNTS.copy(),
            actions=ZERO_ACTIONS.copy(),
        )
        evaluations.append(evaluation)

        image_meta = metadata_map.get(image_path)
        xmp_meta = metadata_map.get(xmp_path)
        image_data, xmp_data = prepare_pair_metadata(
            image_meta or {}, xmp_meta or {}, image_path, xmp_path, logger
        )
        image_subjects = image_data["subjects"]
        image_subjects_for_compare = image_data["compare_subjects"]
        image_regions = image_data["regions"]
        xmp_subjects = xmp_data["subjects"]
        xmp_regions = xmp_data["regions"]
        base_counts = dict(
            ZERO_COUNTS,
            image_subjects=len(image_subjects),
            image_regions=len(image_regions),
            xmp_subjects=len(xmp_subjects),
            xmp_regions=len(xmp_regions),
        )
        evaluation.counts = base_counts.copy()
        missing_parts = [
            label
            for label, meta in (("image metadata", image_meta), ("sidecar metadata", xmp_meta))
            if meta is None
        ]
        if missing_parts:
            evaluation.reasons.append(REASON_MISSING_METADATA)
            logger.warning(
                "Skipped %s: missing Pass 1 %s.", image_path, " and ".join(missing_parts)
            )
            continue

        if not (xmp_subjects or xmp_regions):
            evaluation.reasons.append(REASON_XMP_EMPTY)
            logger.warning(
                "Skipped %s: XMP sidecar contains no usable Subject or RegionInfo after "
                "normalization.",
                image_path,
            )
            continue

        if mode == MODE_FILL and (image_subjects or image_regions):
            evaluation.reasons.append(REASON_IMAGE_HAS_EXISTING_METADATA)
            logger.warning(
                "Skipped %s: image already contains normalized Subject or RegionInfo metadata.",
                image_path,
            )
            continue

        if mode == MODE_MERGE:
            final_regions, actions, reasons = merge_regions(xmp_regions, image_regions)
            for reason in reasons:
                if reason not in evaluation.reasons:
                    evaluation.reasons.append(reason)
        else:
            final_regions = [copy.deepcopy(region) for region in xmp_regions]
            actions = ZERO_ACTIONS.copy()
            actions["xmp_regions_added"] = actions["xmp_regions_from_xmp"] = len(final_regions)
            if mode == MODE_REPLACE:
                actions["image_regions_removed"] = len(image_regions)

        final_subjects = build_final_subjects(mode, image_subjects, xmp_subjects, final_regions)
        final_region_info: dict[str, Any] = {}
        if dimensions := xmp_data["dimensions"] or image_data["dimensions"]:
            final_region_info["AppliedToDimensions"] = copy.deepcopy(dimensions)
        if final_regions:
            final_region_info["RegionList"] = copy.deepcopy(final_regions)
        evaluation.counts = dict(
            base_counts,
            final_subjects=len(final_subjects),
            final_regions=len(final_regions),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "--- FINAL PREPARED REGION DICTIONARY FOR %s ---\n%s\n"
                "---------------------------------------------------------",
                os.path.basename(image_path),
                json.dumps(final_region_info, indent=2, ensure_ascii=False),
            )

        if (
            image_subjects_for_compare == final_subjects
            and image_data["compare_region_info"] == final_region_info
        ):
            evaluation.decision = DECISION_UNCHANGED
            evaluation.reasons.append(REASON_NO_CHANGE)
            logger.info("Unchanged after evaluation: %s", image_path)
            continue

        evaluation.decision = DECISION_UPDATE
        actions["unnamed_regions_written"] = sum(
            1
            for region in final_regions
            if region_is_face_like(region) and region_real_name(region) is None
        )
        evaluation.actions = actions
        region_info_action = (
            REGION_INFO_WRITE
            if final_region_info
            else REGION_INFO_KEEP if mode == MODE_MERGE else REGION_INFO_CLEAR
        )
        write_tasks.append(
            WriteTask(
                image_path,
                final_subjects,
                region_info_action,
                final_region_info or None,
            )
        )

    return write_tasks, evaluations

def write_pass(
    write_tasks: Sequence[WriteTask],
    keep_original: bool,
    logger: logging.Logger,
) -> tuple[bool, dict[str, str], str | None]:
    """Write metadata in bulk and return per-file outcomes for the audit log."""
    if not write_tasks:
        return True, {}, None

    status_specs = (
        (WRITE_RESULT_FAILED, "-efile", "errors.txt", logger.error, "Failed to update: %s"),
        (WRITE_RESULT_UNCHANGED, "-efile2", "unchanged.txt", logger.info, "Unchanged: %s"),
        (WRITE_RESULT_UPDATED, "-efile8", "updated.txt", logger.info, "Processed / Updated: %s"),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        write_args_file = os.path.join(temp_dir, "write_args.txt")
        status_paths = {
            result: os.path.join(temp_dir, filename)
            for result, _, filename, _, _ in status_specs
        }
        for path in status_paths.values():
            open(path, "w", encoding="utf-8").close()

        with open(write_args_file, "w", encoding="utf-8", newline="\n") as handle:
            for task in write_tasks:
                handle.write("-Subject=\n")
                for subject in task.subjects:
                    handle.write(f"-Subject={subject}\n")
                if task.region_info_action == REGION_INFO_WRITE:
                    encoded_region_info = encode_exiftool_struct(task.region_info or {})
                    logger.debug(
                        "--- EXIFTOOL STRUCT STRING ---\n%s\n------------------------------",
                        encoded_region_info,
                    )
                    handle.write(f"-RegionInfo={encoded_region_info}\n")
                elif task.region_info_action == REGION_INFO_CLEAR:
                    handle.write("-RegionInfo=\n")
                for result_key, flag, _, _, _ in status_specs:
                    handle.write(f"{flag}\n{status_paths[result_key]}\n")
                if not keep_original:
                    handle.write("-overwrite_original\n")
                handle.write(f"{task.image_path}\n-execute\n")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- CONTENTS OF WRITE ARGUMENT FILE ---")
            with open(write_args_file, "r", encoding="utf-8") as debug_handle:
                logger.debug("\n%s---------------------------------------", debug_handle.read())

        result = run_exiftool(
            build_exiftool_command(write_args_file, UTF8_CHARSET_ARGS, UTF8_CHARSET_ARGS),
            logger,
            "Pass 2",
        )
        if result is None:
            return (
                False,
                {task.image_path: WRITE_RESULT_FAILED for task in write_tasks},
                "exiftool_execution_failed",
            )

        status_sets: dict[str, set[str]] = {}
        for result_key, _, _, _, _ in status_specs:
            with open(status_paths[result_key], "r", encoding="utf-8", errors="replace") as handle:
                status_sets[result_key] = {
                    os.path.abspath(line.strip()) for line in handle if line.strip()
                }
        expected_paths = {task.image_path for task in write_tasks}
        for path in sorted(set().union(*status_sets.values()) - expected_paths):
            logger.warning("ExifTool reported status for unexpected file: %s", path)

        per_file_results: dict[str, str] = {}
        for task in write_tasks:
            image_path = task.image_path
            for result_key, _, _, log_fn, message in status_specs:
                if image_path in status_sets[result_key]:
                    per_file_results[image_path] = result_key
                    log_fn(message, image_path)
                    break
            else:
                per_file_results[image_path] = WRITE_RESULT_FAILED
                logger.warning("No explicit ExifTool write status returned for: %s", image_path)

        stderr_text = result.stderr.strip()
        if stderr_text:
            if result.returncode == 0:
                logger.warning("ExifTool write reported warnings:\n%s", stderr_text)
            else:
                logger.error("ExifTool write error output:\n%s", stderr_text)
        if result.returncode != 0:
            logger.error("ExifTool write exited with status %s.", result.returncode)

        counts = Counter(per_file_results.values())
        success = result.returncode == 0 and counts[WRITE_RESULT_FAILED] == 0
        if success:
            logger.info(
                "Success! %d file(s) updated, %d unchanged.",
                counts[WRITE_RESULT_UPDATED],
                counts[WRITE_RESULT_UNCHANGED],
            )
            return True, per_file_results, None

        logger.error(
            "Completed with issues: %d updated, %d unchanged, %d failed.",
            counts[WRITE_RESULT_UPDATED],
            counts[WRITE_RESULT_UNCHANGED],
            counts[WRITE_RESULT_FAILED],
        )
        return False, per_file_results, "exiftool_write_failed"

def build_run_audit(
    timestamp: str,
    directory: str,
    mode: str,
    dry_run: bool,
    keep_original: bool,
    pairs_found: int,
    evaluations: Sequence[EvaluatedFile],
    *,
    fatal_error: str | None = None,
    debug_log_path: str | None = None,
) -> dict[str, Any]:
    """Build the structured JSON audit artifact for this run."""
    decision_counts = Counter(evaluation.decision for evaluation in evaluations)
    write_counts = Counter(evaluation.write_result for evaluation in evaluations)
    audit: dict[str, Any] = {
        "run": {
            "timestamp": timestamp,
            "directory": directory,
            "mode": mode,
            "dry_run": dry_run,
            "keep_original": keep_original,
        },
        "summary": {
            "pairs_found": pairs_found,
            "files_evaluated": len(evaluations),
            "files_skipped": decision_counts[DECISION_SKIP],
            "files_planned_for_write": decision_counts[DECISION_UPDATE],
            "files_updated": write_counts[WRITE_RESULT_UPDATED],
            "files_unchanged": (
                decision_counts[DECISION_UNCHANGED] + write_counts[WRITE_RESULT_UNCHANGED]
            ),
            "files_failed": write_counts[WRITE_RESULT_FAILED],
        },
        "files": [dict(vars(evaluation)) for evaluation in evaluations],
    }
    if fatal_error is not None:
        audit["fatal_error"] = fatal_error
    if debug_log_path is not None:
        audit["debug_log_path"] = debug_log_path
    return audit

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed face metadata from XMP sidecars to image files using ExifTool."
    )
    parser.add_argument("directory", help="Target root directory to scan.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=MODE_FILL,
        help="Writing mode to use. Defaults to fill.",
    )
    for flag, help_text in (
        ("--dry-run", "Evaluate and log planned changes without modifying image files."),
        ("--keep-original", "Keep original backup files (do not use -overwrite_original)."),
        ("--debug", "Save verbose debugging output to a separate log file."),
    ):
        parser.add_argument(flag, action="store_true", help=help_text)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = os.path.join("logs", f"face_embed_run_{timestamp}.json")
    logger, debug_file = setup_logger(args.debug, timestamp)

    directory = os.path.abspath(args.directory)
    pairs: list[tuple[str, str]] = []
    evaluations: list[EvaluatedFile] = []
    exit_code = 0
    fatal_error: str | None = None

    logger.info("Starting Face Embedder run. Directory: %s", directory)
    logger.info("Mode: %s", args.mode)
    if args.dry_run:
        logger.info("Dry-run mode enabled. No image files will be modified.")
    if debug_file:
        logger.info("Debug mode enabled. Detailed debug info will be saved to: %s", debug_file)

    try:
        pairs = find_image_xmp_pairs(directory)
        logger.info("Found %d image/xmp pairs to evaluate.", len(pairs))

        if pairs:
            logger.info("Pass 1: Reading existing metadata from files (this may take a moment)...")
            metadata_map = read_pass(pairs, logger)
            if metadata_map is None:
                fatal_error, exit_code = "read_pass_failed", 1
            else:
                write_tasks, evaluations = prepare_write_tasks(
                    pairs, metadata_map, args.mode, logger
                )

                if args.dry_run:
                    planned_writes = 0
                    for evaluation in evaluations:
                        if evaluation.decision == DECISION_UPDATE:
                            evaluation.write_result = WRITE_RESULT_DRY_RUN
                            planned_writes += 1
                    logger.info("Dry-run completed. Planned writes: %d.", planned_writes)
                elif not write_tasks:
                    logger.info("No files require modification after evaluation.")
                else:
                    logger.info("Pass 2: Applying updates to %d files...", len(write_tasks))
                    success, per_file_results, write_error = write_pass(
                        write_tasks, keep_original=args.keep_original, logger=logger
                    )
                    for evaluation in evaluations:
                        if evaluation.decision != DECISION_UPDATE:
                            continue
                        per_file_status = per_file_results.get(evaluation.image_path)
                        if per_file_status is None:
                            evaluation.write_result = WRITE_RESULT_FAILED
                            evaluation.error = "missing_write_result"
                            continue
                        evaluation.write_result = per_file_status
                        if per_file_status == WRITE_RESULT_FAILED and evaluation.error is None:
                            evaluation.error = write_error or "write_failed"
                    if success:
                        logger.info("Run completed successfully.")
                    else:
                        exit_code = 1
                        logger.error("Run completed with errors.")
        else:
            logger.info("No matching pairs found.")
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        fatal_error = str(exc)
        exit_code = 1
        logger.exception("Unhandled error during run: %s", exc)
    finally:
        audit = build_run_audit(
            timestamp,
            directory,
            args.mode,
            args.dry_run,
            args.keep_original,
            len(pairs),
            evaluations,
            fatal_error=fatal_error,
            debug_log_path=debug_file,
        )
        try:
            os.makedirs(os.path.dirname(audit_path), exist_ok=True)
            with open(audit_path, "w", encoding="utf-8") as handle:
                json.dump(audit, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            logger.info("Wrote run audit JSON: %s", audit_path)
        except OSError as exc:
            logger.error("Failed to write run audit JSON %s: %s", audit_path, exc)
            exit_code = 1

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
