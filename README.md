# Face Metadata Embedder

Embed existing face metadata from matching `.xmp` sidecar files into image files with ExifTool.

This script does **not** detect faces or run face recognition. It only reads existing `Subject` and `RegionInfo` metadata from the image and sidecar, resolves it according to the selected mode, and writes the result back to the image.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

Sidecars are matched recursively by image filename plus `.xmp` (case-insensitive). Example:

- `photo.jpg`
- `photo.jpg.xmp`

## Requirements

- Python 3.10+
- ExifTool available on `PATH`

Examples:

```bash
# macOS
brew install exiftool

# Ubuntu / Debian
sudo apt-get install libimage-exiftool-perl
```

## Usage

```bash
python embed_faces.py /path/to/photos
python embed_faces.py /path/to/photos --mode merge
python embed_faces.py /path/to/photos --mode replace --dry-run
```

Options:

- `--mode {fill,merge,replace}`: write mode. Default: `fill`
- `--dry-run`: evaluate and log planned changes without writing files
- `--keep-original`: keep ExifTool `_original` backups
- `--debug`: write a verbose debug log

## How face data is handled

The script only instructs ExifTool to read and write `Subject` and `RegionInfo`.

For face-like regions (blank `Type` or `Face`):

- blank names are treated as unnamed
- the literal `Unknown` is treated as unnamed
- unnamed face boxes are kept in `RegionInfo`
- unnamed faces are excluded from `Subject`

`Subject` is rewritten from the resolved result whenever a file is updated:

- in `merge`, existing image subjects are kept, then combined with sidecar subjects and names from the final resolved regions
- in `fill` and `replace`, existing image subjects are not preserved
- values are deduplicated case-insensitively

This matters if you use `Subject` for non-face keywords: the script does not distinguish face names from other `Subject` entries.

## Modes

### `fill`

Default mode.

The image is only updated when it has no usable normalized `Subject` values and no `RegionInfo` regions already present. In practice, it skips files that already contain either:

- at least one normalized `Subject` value
- at least one `RegionInfo` region

A normalized `Subject` value means a non-blank value other than the literal `Unknown`. This includes generic keywords.

### `merge`

Sidecar regions are used as the base result.

Then existing image regions are compared against sidecar regions:

- non-overlapping image regions are kept
- overlapping regions with the same real name collapse to one region
- if one overlapping region is named and the other is unnamed, the named one wins
- if overlapping named regions have different real names, both are kept and the audit log records `overlap_different_real_names`

Existing image `Subject` values are preserved in this mode.

Note that sidecar regions are deduplicated by overlap before merge. Existing image regions are not globally deduplicated against each other.

### `replace`

Existing image face metadata is replaced by the sidecar-derived result.

- existing image regions are removed
- existing image `Subject` values are not preserved
- final `Subject` and `RegionInfo` come from the normalized sidecar data and resolved region names

## Logging

Each run writes a JSON audit log to `logs/`:

```text
logs/face_embed_run_YYYYMMDD_HHMMSS.json
```

With `--debug`, the script also writes a text debug log:

```text
logs/face_embed_debug_YYYYMMDD_HHMMSS.txt
```

The audit log includes per-file decisions such as:

- `missing_metadata`
- `xmp_empty`
- `image_has_existing_metadata`
- `no_change`
- `overlap_different_real_names`

## Notes

- The script scans directories recursively.
- If a sidecar has no usable `Subject` or `RegionInfo` after normalization, that pair is skipped.
- `AppliedToDimensions` is taken from the sidecar when present, otherwise from the image metadata.
