#!/usr/bin/env python3
"""Embed face metadata from .xmp sidecars into image files using ExifTool.

Requires Python 3.7+.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

LOGGER_NAME = "FaceEmbedder"
EXIFTOOL_EXECUTABLE = "exiftool"
MIN_PYTHON = (3, 7)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# RegionInfo payload semantics in WriteTask:
# - dict with keys: write this RegionInfo structure
# - empty dict: clear RegionInfo explicitly
# - None: leave RegionInfo untouched
WriteTask = Tuple[str, List[str], Optional[Dict[str, Any]]]
MetadataMap = Dict[str, Dict[str, Any]]


def setup_logger(debug_mode: bool) -> Tuple[logging.Logger, str, Optional[str]]:
    """Set up console/file logging without duplicating handlers across runs."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_filename = os.path.join(log_dir, f"face_embed_log_{timestamp}.txt")
    debug_log_filename: Optional[str] = None

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    main_handler = logging.FileHandler(main_log_filename, encoding="utf-8")
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(formatter)
    logger.addHandler(main_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    if debug_mode:
        debug_log_filename = os.path.join(log_dir, f"face_embed_debug_{timestamp}.txt")
        debug_handler = logging.FileHandler(debug_log_filename, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

    return logger, main_log_filename, debug_log_filename


def ensure_supported_python() -> None:
    """Exit early with a clear message on unsupported Python versions."""
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        raise SystemExit(f"This script requires Python {required} or newer.")


def find_image_xmp_pairs(root_dir: str) -> List[Tuple[str, str]]:
    """Recursively find images that have an exact matching .xmp sidecar."""
    pairs: List[Tuple[str, str]] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            image_path = os.path.abspath(os.path.join(dirpath, filename))
            xmp_path = image_path + ".xmp"
            if os.path.exists(xmp_path):
                pairs.append((image_path, os.path.abspath(xmp_path)))

    return pairs


def normalize_list(data: Any) -> List[Any]:
    """Normalize ExifTool output into a flat Python list."""
    if data is None:
        return []
    if isinstance(data, list):
        return list(data)
    if isinstance(data, tuple):
        return list(data)
    if isinstance(data, set):
        return list(data)
    return [data]


def normalize_subjects(data: Any) -> List[str]:
    """Normalize Subject values while preserving order."""
    subjects: List[str] = []
    for item in normalize_list(data):
        if item is None:
            continue
        subjects.append(str(item))
    return subjects


def normalize_region_info(region_info: Any, source_path: str) -> Dict[str, Any]:
    """Normalize RegionInfo into a predictable internal dictionary shape.

    Returns an empty dict when RegionInfo is missing.
    Raises ValueError when the structure exists but is malformed.
    """
    if region_info is None:
        return {}

    if not isinstance(region_info, dict):
        raise ValueError(f"Invalid RegionInfo in {source_path}: expected an object.")

    normalized: Dict[str, Any] = {}

    applied_dims = region_info.get("AppliedToDimensions")
    if applied_dims is not None:
        if not isinstance(applied_dims, dict):
            raise ValueError(
                f"Invalid RegionInfo.AppliedToDimensions in {source_path}: expected an object."
            )
        if applied_dims:
            normalized["AppliedToDimensions"] = applied_dims

    raw_regions = region_info.get("RegionList")
    regions: List[Dict[str, Any]] = []
    for index, region in enumerate(normalize_list(raw_regions)):
        if not isinstance(region, dict):
            raise ValueError(
                f"Invalid RegionInfo.RegionList[{index}] in {source_path}: expected an object."
            )
        regions.append(region)

    if regions:
        normalized["RegionList"] = regions

    return normalized


def safe_normalize_region_info(
    region_info: Any,
    source_path: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Normalize RegionInfo, but degrade malformed data to a warning and fallback."""
    try:
        return normalize_region_info(region_info, source_path)
    except ValueError as exc:
        logger.warning("Ignoring malformed RegionInfo in %s: %s", source_path, exc)
        return {}


def filter_unknown_faces(
    subjects: Sequence[str],
    regions: Sequence[Dict[str, Any]],
    include_unknown: bool,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Filter out explicit 'Unknown' names unless requested."""
    if include_unknown:
        return list(subjects), list(regions)

    filtered_subjects = [
        subject for subject in subjects if subject.strip().lower() != "unknown"
    ]

    filtered_regions: List[Dict[str, Any]] = []
    for region in regions:
        region_name = str(region.get("Name", "")).strip().lower()
        if region_name != "unknown":
            filtered_regions.append(region)

    return filtered_subjects, filtered_regions


def is_duplicate_region(reg1: Dict[str, Any], reg2: Dict[str, Any]) -> bool:
    """Check whether two face regions are effectively identical."""
    if reg1.get("Name") != reg2.get("Name"):
        return False

    area1 = reg1.get("Area") if isinstance(reg1.get("Area"), dict) else {}
    area2 = reg2.get("Area") if isinstance(reg2.get("Area"), dict) else {}

    for key in ["X", "Y", "W", "H"]:
        try:
            value1 = float(area1.get(key, 0))
            value2 = float(area2.get(key, 0))
        except (TypeError, ValueError):
            return False

        if abs(value1 - value2) > 0.001:
            return False

    return True


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
    base_args: Optional[Sequence[str]] = None,
    common_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build an ExifTool command for an argfile.

    ``-charset filename=utf8`` must appear before ``-@`` so UTF-8 encoded
    argfiles work reliably with Unicode filenames on Windows. When the argfile
    contains ``-execute``, ``-common_args`` should repeat any charset options
    that must remain active for every executed command.
    """
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
) -> Optional[subprocess.CompletedProcess]:
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
    expected_files: Set[str],
    logger: logging.Logger,
) -> Optional[MetadataMap]:
    """Build a SourceFile -> metadata map from Pass 1 output.

    Structural problems in the JSON remain fatal, but missing or unexpected file
    coverage is downgraded to warnings so the rest of the batch can continue.
    """
    if not isinstance(raw_metadata, list):
        logger.error("Pass 1 did not return a JSON array as expected.")
        return None

    source_paths: List[str] = []
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

    duplicate_paths = [
        path for path, count in Counter(source_paths).items() if count > 1
    ]
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
    pairs: Sequence[Tuple[str, str]],
    logger: logging.Logger,
) -> Optional[MetadataMap]:
    """Read metadata for all image/xmp pairs in one ExifTool pass."""
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
            base_args=[
                "-q",             # Silence summary messages (e.g., "8 image files read")
                "-charset",       # Set internal text encoding...
                "utf8",           # ...to UTF-8 to correctly handle names with special characters
                "-charset",       # Set file path encoding...
                "filename=utf8",  # ...to UTF-8 so folders/files with Unicode characters don't crash
                "-j",             # Format the output as a JSON string so Python can parse it
                "-struct",        # Keep complex data (like face coordinates) as nested JSON objects
                "-Subject",       # Limit extraction to ONLY the <dc:subject> tag
                "-RegionInfo",    # Limit extraction to ONLY the Face Regions tag
            ],
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
                "ExifTool read exited with status %s; continuing with returned JSON and skipping any missing pair(s).",
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
            raw_metadata = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Pass 1 JSON output: %s", exc)
            return None

        logger.debug(
            "--- RAW READ METADATA ---\n%s\n-------------------------",
            json.dumps(raw_metadata, indent=2, ensure_ascii=False),
        )

        return build_metadata_map(raw_metadata, expected_files, logger)
    finally:
        if os.path.exists(read_args_file):
            os.remove(read_args_file)


def prepare_write_tasks(
    pairs: Sequence[Tuple[str, str]],
    metadata_map: MetadataMap,
    merge: bool,
    include_unknown: bool,
    logger: logging.Logger,
) -> List[WriteTask]:
    """Evaluate metadata and prepare final write operations."""
    write_tasks: List[WriteTask] = []

    for image_path, xmp_path in pairs:
        image_meta = metadata_map.get(image_path)
        xmp_meta = metadata_map.get(xmp_path)

        missing_parts: List[str] = []
        if image_meta is None:
            missing_parts.append("image metadata")
        if xmp_meta is None:
            missing_parts.append("sidecar metadata")
        if missing_parts:
            logger.warning(
                "Skipped %s: missing Pass 1 %s.",
                image_path,
                " and ".join(missing_parts),
            )
            continue

        xmp_subjects = normalize_subjects(xmp_meta.get("Subject"))
        image_subjects = normalize_subjects(image_meta.get("Subject"))
        xmp_region_info = safe_normalize_region_info(
            xmp_meta.get("RegionInfo"), xmp_path, logger
        )
        image_region_info = safe_normalize_region_info(
            image_meta.get("RegionInfo"), image_path, logger
        )

        xmp_regions = list(xmp_region_info.get("RegionList", []))
        image_regions = list(image_region_info.get("RegionList", []))

        xmp_subjects, xmp_regions = filter_unknown_faces(
            xmp_subjects, xmp_regions, include_unknown
        )

        if not xmp_subjects and not xmp_regions:
            logger.warning(
                "Skipped %s: XMP sidecar contains no valid Subject or RegionInfo "
                "(or they were ignored as 'Unknown').",
                image_path,
            )
            continue

        if (image_subjects or image_regions) and not merge:
            logger.warning(
                "Skipped %s: Image already contains metadata (Subject or Regions) "
                "and --merge flag is off.",
                image_path,
            )
            continue

        for region in xmp_regions:
            if region.get("Type") == "Face" and region.get("Name"):
                name = str(region.get("Name")).strip()
                if name and name not in xmp_subjects:
                    xmp_subjects.append(name)

        final_subjects = list(image_subjects) if merge else []
        for subject in xmp_subjects:
            if subject not in final_subjects:
                final_subjects.append(subject)

        final_regions = list(image_regions) if merge else []
        for candidate_region in xmp_regions:
            duplicate_found = any(
                is_duplicate_region(candidate_region, existing_region)
                for existing_region in final_regions
            )
            if not duplicate_found:
                final_regions.append(candidate_region)

        final_region_info: Dict[str, Any] = {}
        dimensions = xmp_region_info.get("AppliedToDimensions")
        if not dimensions:
            dimensions = image_region_info.get("AppliedToDimensions")
        if dimensions:
            final_region_info["AppliedToDimensions"] = dimensions
        if final_regions:
            final_region_info["RegionList"] = final_regions

        logger.debug(
            "--- FINAL PREPARED REGION DICTIONARY FOR %s ---\n%s\n"
            "---------------------------------------------------------",
            os.path.basename(image_path),
            json.dumps(final_region_info, indent=2, ensure_ascii=False),
        )

        region_info_payload: Optional[Dict[str, Any]]
        if final_region_info:
            region_info_payload = final_region_info
        elif merge:
            region_info_payload = None
        else:
            region_info_payload = {}

        write_tasks.append((image_path, final_subjects, region_info_payload))

    return write_tasks


def read_status_file(path: str) -> Set[str]:
    """Read ExifTool status files created via -efile* options."""
    if not os.path.exists(path):
        return set()

    statuses: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                statuses.add(os.path.abspath(stripped))
    return statuses


def write_pass(
    write_tasks: Sequence[WriteTask],
    keep_original: bool,
    logger: logging.Logger,
) -> bool:
    """Write metadata in bulk and report per-file success/failure accurately."""
    with tempfile.TemporaryDirectory() as temp_dir:
        write_args_file = os.path.join(temp_dir, "write_args.txt")
        errors_file = os.path.join(temp_dir, "errors.txt")
        unchanged_file = os.path.join(temp_dir, "unchanged.txt")
        updated_file = os.path.join(temp_dir, "updated.txt")

        for path in [errors_file, unchanged_file, updated_file]:
            with open(path, "w", encoding="utf-8"):
                pass

        with open(write_args_file, "w", encoding="utf-8", newline="\n") as handle:
            for image_path, subjects, region_info in write_tasks:
                
                # ExifTool flag to clear all existing Subject values explicitly before inserting new ones.
                handle.write("-Subject=\n")
                
                # Appends each new subject value one by one to a list tag. 
                for subject in subjects:
                    handle.write(f"-Subject={subject}\n")

                if region_info is None:
                    # 'None' means we are merging and don't want to interfere with RegionInfo
                    pass
                elif region_info:
                    # Write complex JSON-like nested object into ExifTool struct formatting
                    encoded_region_info = encode_exiftool_struct(region_info)
                    logger.debug(
                        "--- EXIFTOOL STRUCT STRING ---\n%s\n------------------------------",
                        encoded_region_info,
                    )
                    handle.write(f"-RegionInfo={encoded_region_info}\n")
                else:
                    # If dict is explicitly empty, we clear the RegionInfo tag entirely
                    handle.write("-RegionInfo=\n")

                # -efile options track success/failure states into separate text files. 
                # This allows Python to read the exact execution results without parsing stdout manually.
                handle.write("-efile\n")
                handle.write(f"{errors_file}\n")    # Logs files that failed/encountered errors
                handle.write("-efile2\n")
                handle.write(f"{unchanged_file}\n") # Logs files evaluated but didn't require saving changes
                handle.write("-efile8\n")
                handle.write(f"{updated_file}\n")   # Logs files successfully modified and saved

                # Flag to prevent ExifTool from duplicating the original file into 'filename.jpg_original'
                if not keep_original:
                    handle.write("-overwrite_original\n")

                # Provide the target image path ExifTool needs to operate on for this block
                handle.write(f"{image_path}\n")
                
                # Resets arguments for the next file iteration and forces immediate execution of the block above
                handle.write("-execute\n")

        logger.debug("--- CONTENTS OF WRITE ARGUMENT FILE ---")
        with open(write_args_file, "r", encoding="utf-8") as debug_handle:
            logger.debug("\n%s---------------------------------------", debug_handle.read())

        command = build_exiftool_command(
            write_args_file,
            base_args=[
                "-charset", "utf8",             # Use UTF-8 for parsing text structures
                "-charset", "filename=utf8"     # Treat filenames as UTF-8 (vital for non-ASCII paths)
            ],
            common_args=[
                "-charset", "utf8",             # -common_args persists across the internal "-execute" resets
                "-charset", "filename=utf8"     # so we don't lose UTF-8 context after the 1st file writes
            ],
        )
        result = run_exiftool(command, logger, "Pass 2")
        if result is None:
            return False

        error_paths = read_status_file(errors_file)
        unchanged_paths = read_status_file(unchanged_file)
        updated_paths = read_status_file(updated_file)
        expected_paths = {image_path for image_path, _, _ in write_tasks}

        unexpected_status_paths = sorted(
            (error_paths | unchanged_paths | updated_paths) - expected_paths
        )
        for path in unexpected_status_paths:
            logger.warning("ExifTool reported status for unexpected file: %s", path)

        updated_count = 0
        unchanged_count = 0
        error_count = 0
        unreported_count = 0

        for image_path, _, _ in write_tasks:
            if image_path in error_paths:
                logger.error("Failed to update: %s", image_path)
                error_count += 1
            elif image_path in updated_paths:
                logger.info("Processed / Updated: %s", image_path)
                updated_count += 1
            elif image_path in unchanged_paths:
                logger.info("Unchanged: %s", image_path)
                unchanged_count += 1
            else:
                logger.warning("No explicit ExifTool write status returned for: %s", image_path)
                unreported_count += 1

        stderr_text = result.stderr.strip()
        if stderr_text:
            if result.returncode == 0:
                logger.warning("ExifTool write reported warnings:\n%s", stderr_text)
            else:
                logger.error("ExifTool write error output:\n%s", stderr_text)

        if result.returncode != 0:
            logger.error("ExifTool write exited with status %s.", result.returncode)

        if error_count == 0 and result.returncode == 0 and unreported_count == 0:
            logger.info(
                "Success! %d file(s) updated, %d unchanged. Check the log for details.",
                updated_count,
                unchanged_count,
            )
            return True

        logger.error(
            "Completed with issues: %d updated, %d unchanged, %d failed, %d unreported. "
            "Check the log for details.",
            updated_count,
            unchanged_count,
            error_count,
            unreported_count,
        )
        return False


def main() -> int:
    ensure_supported_python()

    parser = argparse.ArgumentParser(
        description="Embed Face metadata from XMP sidecars to Images using ExifTool."
    )
    parser.add_argument("directory", help="Target root directory to scan.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge new XMP tags with existing image tags instead of skipping.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original backup files (do not use -overwrite_original).",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include face regions and subjects named 'Unknown' (they are ignored by default).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save verbose debugging output to a separate log file.",
    )
    args = parser.parse_args()

    logger, log_file, debug_file = setup_logger(args.debug)
    logger.info("Starting Face Embedder run. Directory: %s", args.directory)

    if args.debug and debug_file:
        logger.info(
            "Debug mode enabled. Detailed debug info will be saved to: %s",
            debug_file,
        )

    pairs = find_image_xmp_pairs(args.directory)
    logger.info("Found %d image/xmp pairs to evaluate.", len(pairs))
    if not pairs:
        logger.info("No matching pairs found. Exiting.")
        return 0

    logger.info("Pass 1: Reading existing metadata from files (this may take a moment)...")
    metadata_map = read_pass(pairs, logger)
    if metadata_map is None:
        return 1

    write_tasks = prepare_write_tasks(
        pairs,
        metadata_map,
        merge=args.merge,
        include_unknown=args.include_unknown,
        logger=logger,
    )
    if not write_tasks:
        logger.info("No files require modification after evaluation.")
        return 0

    logger.info("Pass 2: Applying updates to %d files...", len(write_tasks))
    success = write_pass(write_tasks,
                         keep_original=args.keep_original,
                         logger=logger)
    if success:
        logger.info("Run completed successfully. Check %s for details.", log_file)
        return 0

    logger.error("Run completed with errors. Check %s for details.", log_file)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
