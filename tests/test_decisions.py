"""Behavioral contract for metadata normalization and fill/merge/replace decisions.

Run from the project directory with: python -m unittest discover -s tests -v
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import embed_faces as app


def face(name=None, *, x=0.5, y=0.5, w=0.2, h=0.2, kind="Face"):
    region = {"Type": kind, "Area": {"X": x, "Y": y, "W": w, "H": h}}
    if name is not None:
        region["Name"] = name
    return region


def metadata(subject=None, regions=None, dimensions=None):
    result = {}
    if subject is not None:
        result["Subject"] = subject
    if regions is not None or dimensions is not None:
        result["RegionInfo"] = {}
        if regions is not None:
            result["RegionInfo"]["RegionList"] = regions
        if dimensions is not None:
            result["RegionInfo"]["AppliedToDimensions"] = dimensions
    return result


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.decisions")
        self.image, self.sidecar = "/photos/a.jpg", "/photos/a.jpg.xmp"

    def decide(self, mode, image=None, sidecar=None):
        meta = {}
        if image is not None:
            meta[self.image] = image
        if sidecar is not None:
            meta[self.sidecar] = sidecar
        tasks, evaluations = app.prepare_write_tasks(
            [(self.image, self.sidecar)], meta, mode, self.logger
        )
        return tasks, evaluations[0]

    def test_pairs_are_recursive_and_match_full_filename_case_insensitively(self):
        with tempfile.TemporaryDirectory() as root:
            nested = Path(root, "subfolder")
            nested.mkdir()
            for name in ("A.JPG", "A.JPG.XMP", "other.png", "other.xmp", "b.tif", "c.txt.xmp"):
                (nested / name).touch()
            self.assertEqual(
                app.find_image_xmp_pairs(root),
                [(str(nested / "A.JPG"), str(nested / "A.JPG.XMP"))],
            )

    def test_all_supported_image_extensions_require_exact_sidecars(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            for extension in ("jpg", "jpeg", "png", "tif", "tiff"):
                (folder / f"person.{extension}").touch()
                (folder / f"person.{extension}.XMP").touch()
            (folder / "unpaired.png").touch()
            (folder / "unpaired.xmp").touch()
            self.assertEqual(
                set(app.find_image_xmp_pairs(root)),
                {
                    (str(folder / f"person.{extension}"), str(folder / f"person.{extension}.XMP"))
                    for extension in ("jpg", "jpeg", "png", "tif", "tiff")
                },
            )

    def test_subjects_trim_unknown_and_deduplicate_without_losing_first_spelling(self):
        self.assertEqual(
            app.normalize_subjects(["  Liis  ", "liis", "UNKNOWN", " ", None, "Märt"]),
            ["Liis", "Märt"],
        )

    def test_face_names_normalize_but_non_face_names_survive(self):
        original = face(" Unknown ")
        self.assertNotIn("Name", app.normalize_region(original))
        self.assertEqual(original["Name"], " Unknown ")
        self.assertEqual(app.normalize_region(face(" Unknown ", kind="Pet"))["Name"], " Unknown ")

    def test_bad_region_info_warns_and_degrades_to_empty(self):
        with self.assertLogs(self.logger, "WARNING") as messages:
            self.assertEqual(app.safe_normalize_region_info({"RegionList": [42]}, "bad.xmp", self.logger), {})
        self.assertIn("bad.xmp", messages.output[0])
        for invalid in ("string", {"AppliedToDimensions": []}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    app.normalize_region_info(invalid, "bad.xmp")

    def test_overlap_uses_iou_or_containment_and_checks_type(self):
        base = face("A")
        # At x=.534 the IoU exceeds .70 while smaller-box coverage is below .85.
        self.assertTrue(app.is_significant_overlap(base, face("B", x=0.534)))
        self.assertFalse(app.is_significant_overlap(base, face("B", x=0.536)))
        self.assertTrue(app.is_significant_overlap(base, face("B", w=0.05, h=0.05)))
        self.assertFalse(app.is_significant_overlap(base, face("B", x=0.9)))
        self.assertFalse(app.is_significant_overlap(base, face("B", kind="Pet")))
        self.assertFalse(app.is_significant_overlap(base, face("B", w=0)))
        self.assertFalse(app.is_significant_overlap(base, {"Area": {"X": "bad"}}))

    def test_overlap_threshold_boundaries_are_inclusive(self):
        base, shifted = face(w=2, h=2, x=0), face(w=2, h=2, x=0.5)
        # Overlap is 75% of either box; IoU is 60%.
        self.assertTrue(app.is_significant_overlap(base, shifted, threshold=0.6))
        self.assertFalse(app.is_significant_overlap(base, shifted, threshold=0.6001))
        self.assertTrue(app.is_significant_overlap(base, shifted, threshold=0.7, containment_threshold=0.75))
        self.assertFalse(app.is_significant_overlap(base, shifted, threshold=0.7, containment_threshold=0.7501))

    def test_vertical_separation_and_partial_overlap_are_respected(self):
        original = face(x=0.5, y=0.5, w=0.15, h=0.3)
        self.assertFalse(app.is_significant_overlap(original, face(x=0.5, y=0.9, w=0.15, h=0.3)))
        self.assertFalse(app.is_significant_overlap(original, face(x=0.5, y=0.64, w=0.15, h=0.3)))
        self.assertTrue(app.is_significant_overlap(original, face(x=0.5, y=0.51, w=0.15, h=0.3)))

    def test_sidecar_dedup_keeps_last_overlapping_region(self):
        first, last, other = face("first"), face("last", x=0.51), face("elsewhere", x=0.9)
        original = [first, other, last]
        self.assertEqual(app.deduplicate_regions(original), [other, last])
        self.assertEqual(original, [first, other, last])

    def test_fill_skips_existing_subject_even_if_it_is_a_generic_keyword(self):
        tasks, result = self.decide(app.MODE_FILL, metadata("holiday"), metadata("Liis"))
        self.assertEqual(tasks, [])
        self.assertEqual(result.reasons, [app.REASON_IMAGE_HAS_EXISTING_METADATA])
        self.assertEqual(result.decision, app.DECISION_SKIP)

    def test_fill_skips_existing_unnamed_face(self):
        tasks, result = self.decide(
            app.MODE_FILL, metadata("Unknown", [face("Unknown")]), metadata("Liis")
        )
        self.assertEqual(tasks, [])
        self.assertEqual(result.counts["image_regions"], 1)
        self.assertEqual(result.reasons, [app.REASON_IMAGE_HAS_EXISTING_METADATA])

    def test_fill_treats_only_unknown_image_subject_as_empty(self):
        tasks, result = self.decide(app.MODE_FILL, metadata([" UNKNOWN "]), metadata("Liis"))
        self.assertEqual(result.decision, app.DECISION_UPDATE)
        self.assertEqual(tasks[0].subjects, ["Liis"])

    def test_missing_metadata_and_empty_sidecar_are_skipped(self):
        _, missing = self.decide(app.MODE_REPLACE, image=metadata("old"))
        _, empty = self.decide(app.MODE_REPLACE, metadata("old"), metadata(["Unknown", " "]))
        self.assertEqual(missing.reasons, [app.REASON_MISSING_METADATA])
        self.assertEqual(empty.reasons, [app.REASON_XMP_EMPTY])

    def test_fill_preserves_unnamed_box_but_does_not_add_unknown_subject(self):
        tasks, result = self.decide(
            app.MODE_FILL, {}, metadata(["Unknown"], [face("Unknown")])
        )
        self.assertEqual(result.decision, app.DECISION_UPDATE)
        self.assertEqual(tasks[0].subjects, [])
        self.assertEqual(tasks[0].region_info_action, app.REGION_INFO_WRITE)
        self.assertNotIn("Name", tasks[0].region_info["RegionList"][0])
        self.assertEqual(result.actions["unnamed_regions_written"], 1)

    def test_merge_preserves_image_subjects_and_distinct_regions(self):
        dims = {"W": 4000, "H": 3000, "Unit": "pixel"}
        tasks, result = self.decide(
            app.MODE_MERGE,
            metadata(["holiday", "Liis"], [face("Märt", x=0.9)], {"W": 10}),
            metadata(["liis", "Unknown"], [face("Liis")], dims),
        )
        self.assertEqual(tasks[0].subjects, ["holiday", "Liis", "Märt"])
        self.assertEqual([r["Name"] for r in tasks[0].region_info["RegionList"]], ["Liis", "Märt"])
        self.assertEqual(tasks[0].region_info["AppliedToDimensions"], dims)
        self.assertEqual(result.actions["image_regions_kept"], 1)

    def test_merge_overlap_name_resolution_and_conflict_audit(self):
        cases = (
            ("Liis", "liis", ["Liis"], 1, 0),
            ("Liis", "Unknown", ["Liis"], 1, 0),
            ("Unknown", "Liis", ["Liis"], 1, 0),
            ("Liis", "Märt", ["Liis", "Märt"], 0, 1),
        )
        for sidecar_name, image_name, names, collapsed, conflicts in cases:
            with self.subTest(sidecar=sidecar_name, image=image_name):
                tasks, result = self.decide(
                    app.MODE_MERGE,
                    metadata(regions=[face(image_name, x=0.51)]),
                    metadata(regions=[face(sidecar_name)]),
                )
                regions = tasks[0].region_info["RegionList"]
                self.assertEqual([r["Name"] for r in regions], names)
                # A resolved face keeps the sidecar box even when its name comes from the image.
                self.assertEqual([r["Area"]["X"] for r in regions], [0.5, 0.51] if conflicts else [0.5])
                self.assertEqual(result.actions["regions_collapsed"], collapsed)
                self.assertEqual(result.actions["conflicts_kept_both"], conflicts)
                self.assertEqual(
                    app.REASON_OVERLAP_DIFFERENT_REAL_NAMES in result.reasons, bool(conflicts)
                )

    def test_overlapping_unnamed_faces_keep_one_sidecar_box(self):
        sidecar_box, image_box = face("Unknown"), face(" ", x=0.51)
        tasks, result = self.decide(
            app.MODE_MERGE, metadata(regions=[image_box]), metadata(regions=[sidecar_box])
        )
        self.assertEqual(tasks[0].subjects, [])
        self.assertEqual(tasks[0].region_info["RegionList"], [face()])
        self.assertEqual(result.actions["regions_collapsed"], 1)
        self.assertEqual(result.actions["unnamed_regions_written"], 1)

    def test_sidecar_regions_deduplicate_during_preparation(self):
        first, last = face("first"), face("last", x=0.51)
        tasks, result = self.decide(
            app.MODE_REPLACE, {}, metadata(regions=[first, last])
        )
        self.assertEqual(tasks[0].region_info["RegionList"], [last])
        self.assertEqual(tasks[0].subjects, ["last"])
        self.assertEqual(result.counts["xmp_regions"], 1)
        self.assertEqual(result.actions["xmp_regions_added"], 1)

    def test_merge_does_not_deduplicate_existing_image_regions_against_each_other(self):
        first, second, sidecar_box = face("first"), face("second", x=0.51), face("sidecar", x=0.9)
        tasks, result = self.decide(
            app.MODE_MERGE, metadata(regions=[first, second]), metadata(regions=[sidecar_box])
        )
        self.assertEqual(tasks[0].region_info["RegionList"], [sidecar_box, first, second])
        self.assertEqual(result.actions["image_regions_kept"], 2)
        self.assertEqual(result.actions["regions_collapsed"], 0)

    def test_blank_type_is_a_face_and_other_types_do_not_add_subjects(self):
        blank_face, pet = face("  Liis  ", kind=" "), face("Cat", kind="Pet", x=0.9)
        tasks, _ = self.decide(app.MODE_REPLACE, {}, metadata(regions=[blank_face, pet]))
        self.assertEqual(tasks[0].subjects, ["Liis"])
        self.assertEqual(tasks[0].region_info["RegionList"], [face("Liis", kind=" "), pet])

        tasks, result = self.decide(
            app.MODE_MERGE,
            metadata(regions=[face("Image", kind="Pet")]),
            metadata(regions=[face("Sidecar", kind="Pet")]),
        )
        self.assertEqual(tasks[0].subjects, [])
        self.assertEqual(tasks[0].region_info["RegionList"], [face("Sidecar", kind="Pet")])
        self.assertEqual(result.actions["image_regions_removed"], 1)

    def test_dimensions_fall_back_to_image_when_sidecar_has_none(self):
        dimensions = {"W": 4000, "H": 3000, "Unit": "pixel"}
        tasks, _ = self.decide(
            app.MODE_REPLACE, metadata(regions=[face("Old")], dimensions=dimensions),
            metadata(regions=[face("Liis")]),
        )
        self.assertEqual(tasks[0].region_info["AppliedToDimensions"], dimensions)
        self.assertEqual(tasks[0].region_info["RegionList"], [face("Liis")])

    def test_replace_drops_old_subjects_and_regions(self):
        tasks, result = self.decide(
            app.MODE_REPLACE,
            metadata(["old"], [face("old")]),
            metadata(["Liis"], [face("Unknown", x=0.9)]),
        )
        self.assertEqual(tasks[0].subjects, ["Liis"])
        self.assertEqual(len(tasks[0].region_info["RegionList"]), 1)
        self.assertEqual(result.actions["image_regions_removed"], 1)

    def test_replace_with_subject_only_explicitly_clears_existing_regions(self):
        tasks, _ = self.decide(
            app.MODE_REPLACE, metadata(regions=[face("old")]), metadata("Liis")
        )
        self.assertEqual(tasks[0].region_info_action, app.REGION_INFO_CLEAR)
        self.assertIsNone(tasks[0].region_info)

    def test_subject_only_modes_choose_keep_or_clear(self):
        merge_tasks, _ = self.decide(
            app.MODE_MERGE, metadata("Old"), metadata("New")
        )
        self.assertEqual(merge_tasks[0].subjects, ["Old", "New"])
        self.assertEqual(merge_tasks[0].region_info_action, app.REGION_INFO_KEEP)
        self.assertIsNone(merge_tasks[0].region_info)

        for mode in (app.MODE_FILL, app.MODE_REPLACE):
            with self.subTest(mode=mode):
                tasks, _ = self.decide(mode, {}, metadata("New"))
                self.assertEqual(tasks[0].subjects, ["New"])
                self.assertEqual(tasks[0].region_info_action, app.REGION_INFO_CLEAR)

    def test_identical_metadata_is_unchanged_and_never_written(self):
        same = metadata(["Liis"], [face("Liis")])
        tasks, result = self.decide(app.MODE_REPLACE, same, same)
        self.assertEqual(tasks, [])
        self.assertEqual(result.decision, app.DECISION_UNCHANGED)
        self.assertEqual(result.reasons, [app.REASON_NO_CHANGE])

    def test_normalized_match_still_rewrites_raw_subjects_and_region_names(self):
        clean = metadata(["Liis"], [face()])
        for dirty in (
            metadata(["  Liis  ", "Unknown"], [face()]),
            metadata(["Liis"], [face("Unknown")]),
        ):
            with self.subTest(dirty=dirty):
                tasks, result = self.decide(app.MODE_REPLACE, dirty, clean)
                self.assertEqual(result.decision, app.DECISION_UPDATE)
                self.assertEqual(tasks[0].subjects, ["Liis"])
                self.assertEqual(tasks[0].region_info["RegionList"], [face()])
                tasks_again, second = self.decide(app.MODE_REPLACE, clean, clean)
                self.assertEqual(tasks_again, [])
                self.assertEqual(second.decision, app.DECISION_UNCHANGED)

    def test_skipped_and_unchanged_pairs_do_not_stop_later_updates(self):
        pairs = [(f"/photos/{name}.jpg", f"/photos/{name}.jpg.xmp") for name in "abcd"]
        values = {
            pairs[1][0]: metadata("Existing"), pairs[1][1]: metadata("Unknown"),
            pairs[2][0]: metadata("Same"), pairs[2][1]: metadata("Same"),
            pairs[3][0]: metadata(), pairs[3][1]: metadata("New"),
        }
        tasks, results = app.prepare_write_tasks(pairs, values, app.MODE_REPLACE, self.logger)
        self.assertEqual([item.decision for item in results], [
            app.DECISION_SKIP, app.DECISION_SKIP, app.DECISION_UNCHANGED, app.DECISION_UPDATE
        ])
        self.assertEqual([item.reasons for item in results[:3]], [
            [app.REASON_MISSING_METADATA], [app.REASON_XMP_EMPTY],
            [app.REASON_NO_CHANGE],
        ])
        self.assertEqual([task.image_path for task in tasks], [pairs[3][0]])

    def test_fill_continues_after_existing_image_metadata(self):
        pairs = [(f"/photos/{name}.jpg", f"/photos/{name}.jpg.xmp") for name in "ab"]
        values = {
            pairs[0][0]: metadata("Existing"), pairs[0][1]: metadata("Sidecar"),
            pairs[1][0]: {}, pairs[1][1]: metadata("New"),
        }
        tasks, results = app.prepare_write_tasks(pairs, values, app.MODE_FILL, self.logger)
        self.assertEqual([result.decision for result in results], [app.DECISION_SKIP, app.DECISION_UPDATE])
        self.assertEqual(results[0].reasons, [app.REASON_IMAGE_HAS_EXISTING_METADATA])
        self.assertEqual([task.image_path for task in tasks], [pairs[1][0]])
        self.assertEqual(tasks[0].subjects, ["New"])


if __name__ == "__main__":
    unittest.main()
