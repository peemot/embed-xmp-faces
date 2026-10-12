"""Tests for ExifTool I/O and CLI decisions using synthetic image data only."""

import json
import hashlib
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import embed_faces as app


def create_synthetic_image_and_sidecar(folder: Path) -> tuple[str, str]:
    """Build a valid blank PNG and MWG face XMP, without any photo fixtures."""
    width, height = 64, 48

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + chunk_type + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress((b"\x00" + b"\xd0\xd0\xd0" * width) * height))
        + png_chunk(b"IEND", b"")
    )
    image = folder / "synthetic.png"
    image.write_bytes(png_data)
    sidecar = folder / "synthetic.png.xmp"
    sidecar.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
      xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#"
      xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
   <dc:subject><rdf:Bag><rdf:li>Synthetic Person</rdf:li></rdf:Bag></dc:subject>
   <mwg-rs:Regions rdf:parseType="Resource">
    <mwg-rs:AppliedToDimensions stDim:w="64" stDim:h="48" stDim:unit="pixel"/>
    <mwg-rs:RegionList>
     <rdf:Bag>
      <rdf:li rdf:parseType="Resource">
       <mwg-rs:Name>Synthetic Person</mwg-rs:Name>
       <mwg-rs:Type>Face</mwg-rs:Type>
       <mwg-rs:Area stArea:x="0.5" stArea:y="0.5"
                    stArea:w="0.25" stArea:h="0.25" stArea:unit="normalized"/>
      </rdf:li>
     </rdf:Bag>
    </mwg-rs:RegionList>
   </mwg-rs:Regions>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )
    return str(image), str(sidecar)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.pipeline")

    def test_synthetic_fixture_has_valid_png_and_face_xmp(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = create_synthetic_image_and_sidecar(Path(root))
            self.assertEqual(Path(image).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            xmp = ElementTree.parse(sidecar)
            face_name = xmp.find(".//{http://www.metadataworkinggroup.com/schemas/regions/}Name")
            self.assertIsNotNone(face_name)
            self.assertEqual(face_name.text, "Synthetic Person")

    def test_metadata_map_rejects_duplicate_and_malformed_results(self):
        image = os.path.abspath("one.jpg")
        for raw in (
            {},
            ["not an object"],
            [{}],
            [{"SourceFile": image}, {"SourceFile": image}],
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(app.build_metadata_map(raw, {image}, self.logger))

    def test_metadata_map_retains_partial_results_and_warns_about_missing(self):
        image = os.path.abspath("one.jpg")
        sidecar = os.path.abspath("one.jpg.xmp")
        with self.assertLogs(self.logger, "WARNING") as logs:
            result = app.build_metadata_map(
                [{"SourceFile": image, "Subject": "Liis"}], {image, sidecar}, self.logger
            )
        self.assertEqual(result[image]["Subject"], "Liis")
        self.assertNotIn(sidecar, result)
        self.assertIn(sidecar, " ".join(logs.output))

    def test_read_pass_sends_both_paths_via_utf8_argfile_and_removes_it(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = (str(Path(root, name)) for name in ("näide.jpg", "näide.jpg.xmp"))
            observed = {}

            def fake_run(command, logger, phase):
                observed["command"] = command[:]
                observed["phase"] = phase
                observed["paths"] = Path(command[command.index("-@") + 1]).read_text(encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0,
                    stdout=json.dumps([{"SourceFile": image}, {"SourceFile": sidecar}]),
                    stderr="",
                )

            with patch.object(app, "run_exiftool", side_effect=fake_run):
                result = app.read_pass([(image, sidecar)], self.logger)
            self.assertEqual(set(result), {image, sidecar})
            self.assertEqual(observed["paths"], f"{image}\n{sidecar}\n")
            self.assertEqual(observed["phase"], "Pass 1")
            self.assertIn("-struct", observed["command"])
            self.assertIn("-RegionInfo", observed["command"])
            self.assertEqual(
                observed["command"][1:6],
                ["-q", "-charset", "utf8", "-charset", "filename=utf8"],
            )
            self.assertFalse(Path(observed["command"][observed["command"].index("-@") + 1]).exists())

    def test_read_pass_accepts_partial_json_after_nonzero_exiftool_exit(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = (str(Path(root, name)) for name in ("a.jpg", "a.jpg.xmp"))
            response = subprocess.CompletedProcess(
                [], 1, stdout=json.dumps([{"SourceFile": image}]), stderr="one file missing"
            )
            with patch.object(app, "run_exiftool", return_value=response):
                with self.assertLogs(self.logger, "WARNING"):
                    result = app.read_pass([(image, sidecar)], self.logger)
            self.assertEqual(set(result), {image})

    def test_read_pass_rejects_failed_or_invalid_output(self):
        with tempfile.TemporaryDirectory() as root:
            pair = [(str(Path(root, "a.jpg")), str(Path(root, "a.jpg.xmp")))]
            for response in (
                None,
                subprocess.CompletedProcess([], 1, stdout="", stderr="failure"),
                subprocess.CompletedProcess([], 0, stdout="not JSON", stderr=""),
            ):
                with self.subTest(response=response):
                    with patch.object(app, "run_exiftool", return_value=response):
                        self.assertIsNone(app.read_pass(pair, self.logger))

    def test_run_exiftool_invokes_subprocess_with_utf8_output(self):
        expected = subprocess.CompletedProcess(["exiftool", "-ver"], 0, "13.0", "")
        with patch.object(app.subprocess, "run", return_value=expected) as run:
            self.assertIs(app.run_exiftool(["exiftool", "-ver"], self.logger, "test"), expected)
        run.assert_called_once_with(
            ["exiftool", "-ver"], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

    def test_run_exiftool_handles_missing_or_unstartable_executable(self):
        for error in (FileNotFoundError(), OSError("cannot start")):
            with self.subTest(error=type(error).__name__):
                with patch.object(app.subprocess, "run", side_effect=error):
                    with self.assertLogs(self.logger, "ERROR"):
                        self.assertIsNone(app.run_exiftool(["exiftool"], self.logger, "test"))

    def test_struct_encoding_escapes_exiftool_metacharacters(self):
        for value, expected in (
            ("Liis|Märt", "Liis||Märt"),
            ("Liis,Märt", "Liis|,Märt"),
            ("Liis}Märt", "Liis|}Märt"),
            ("Liis]Märt", "Liis|]Märt"),
            (" [Õun", "| [Õun"),
            ("[Õun", "|[Õun"),
        ):
            with self.subTest(value=value):
                self.assertEqual(app.encode_exiftool_struct({"Name": value}), f"{{Name={expected}}}")

    def test_write_pass_serializes_independent_tasks_and_explicit_clear(self):
        with tempfile.TemporaryDirectory() as root:
            image1, image2 = (str(Path(root, name)) for name in ("a.jpg", "näide.jpg"))
            tasks = [
                app.WriteTask(
                    image1, ["Liis", "Märt"], app.REGION_INFO_WRITE,
                    {"RegionList": [{"Name": "Liis", "Area": {"X": "0,5"}}]},
                ),
                app.WriteTask(image2, ["new"], app.REGION_INFO_CLEAR),
            ]
            observed = {}

            def fake_run(command, logger, phase):
                argfile = Path(command[command.index("-@") + 1])
                observed["args"] = argfile.read_text(encoding="utf-8").splitlines()
                observed["command"] = command[:]
                observed["phase"] = phase
                for flag, path in (("-efile8", image1), ("-efile2", image2)):
                    index = observed["args"].index(flag)
                    with open(observed["args"][index + 1], "a", encoding="utf-8") as handle:
                        handle.write(path + "\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch.object(app, "run_exiftool", side_effect=fake_run):
                success, results, error = app.write_pass(tasks, keep_original=False, logger=self.logger)
            self.assertTrue(success)
            self.assertIsNone(error)
            self.assertEqual(results, {image1: app.WRITE_RESULT_UPDATED, image2: app.WRITE_RESULT_UNCHANGED})
            self.assertEqual(observed["phase"], "Pass 2")
            self.assertEqual(observed["args"].count("-execute"), 2)
            self.assertEqual(observed["args"].count("-Subject="), 2)
            self.assertEqual(observed["args"].count("-RegionInfo="), 1)
            self.assertIn("-RegionInfo={RegionList=[{Name=Liis,Area={X=0|,5}}]}", observed["args"])
            self.assertEqual(observed["args"].count("-overwrite_original"), 2)
            self.assertIn("-common_args", observed["command"])
            self.assertEqual(observed["command"][1:5], ["-charset", "utf8", "-charset", "filename=utf8"])
            self.assertEqual(observed["command"][-5:],
                             ["-common_args", "-charset", "utf8", "-charset", "filename=utf8"])
            # Every file's values and target must be in its own ExifTool -execute block.
            first_end = observed["args"].index("-execute")
            first = observed["args"][:first_end]
            second = observed["args"][first_end + 1:-1]
            self.assertEqual(first[-1], image1)
            self.assertEqual(second[-1], image2)
            self.assertEqual([value for value in first if value.startswith("-Subject=")],
                             ["-Subject=", "-Subject=Liis", "-Subject=Märt"])
            self.assertEqual([value for value in second if value.startswith("-Subject=")],
                             ["-Subject=", "-Subject=new"])
            self.assertIn("-RegionInfo={RegionList=[{Name=Liis,Area={X=0|,5}}]}", first)
            self.assertNotIn("-RegionInfo=", first)
            self.assertIn("-RegionInfo=", second)
            self.assertFalse(any(value.startswith("-RegionInfo={") for value in second))
            for flag, filename in (
                ("-efile", "errors.txt"),
                ("-efile2", "unchanged.txt"),
                ("-efile8", "updated.txt"),
            ):
                self.assertEqual(first.count(flag), 1)
                self.assertEqual(second.count(flag), 1)
                self.assertEqual(Path(first[first.index(flag) + 1]).name, filename)
                self.assertEqual(second[second.index(flag) + 1], first[first.index(flag) + 1])

    def test_keep_original_does_not_request_overwrite(self):
        image = os.path.abspath("a.jpg")

        def fake_run(command, logger, phase):
            lines = Path(command[command.index("-@") + 1]).read_text(encoding="utf-8").splitlines()
            self.assertNotIn("-overwrite_original", lines)
            self.assertFalse(any(line.startswith("-RegionInfo=") for line in lines))
            with open(lines[lines.index("-efile8") + 1], "w", encoding="utf-8") as handle:
                handle.write(image + "\n")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(app, "run_exiftool", side_effect=fake_run):
            success, results, _ = app.write_pass(
                [app.WriteTask(image, ["Liis"], app.REGION_INFO_KEEP)], True, self.logger
            )
        self.assertTrue(success)
        self.assertEqual(results[image], app.WRITE_RESULT_UPDATED)

    def test_write_pass_marks_unreported_files_failed_and_preserves_reported_successes(self):
        image1, image2 = (os.path.abspath(name) for name in ("a.jpg", "b.jpg"))

        def fake_run(command, logger, phase):
            lines = Path(command[command.index("-@") + 1]).read_text(encoding="utf-8").splitlines()
            with open(lines[lines.index("-efile8") + 1], "w", encoding="utf-8") as handle:
                handle.write(image1 + "\n")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failure")

        tasks = [app.WriteTask(path, [], app.REGION_INFO_CLEAR) for path in (image1, image2)]
        with patch.object(app, "run_exiftool", side_effect=fake_run):
            success, results, error = app.write_pass(tasks, False, self.logger)
        self.assertFalse(success)
        self.assertEqual(error, "exiftool_write_failed")
        self.assertEqual(results, {image1: app.WRITE_RESULT_UPDATED, image2: app.WRITE_RESULT_FAILED})

    def test_write_fails_for_nonzero_exit_even_if_every_file_reports_updated(self):
        image = os.path.abspath("a.jpg")

        def fake_run(command, logger, phase):
            args = Path(command[command.index("-@") + 1]).read_text(encoding="utf-8").splitlines()
            Path(args[args.index("-efile8") + 1]).write_text(image + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed after update")

        with patch.object(app, "run_exiftool", side_effect=fake_run):
            success, results, error = app.write_pass(
                [app.WriteTask(image, ["Liis"], app.REGION_INFO_KEEP)], False, self.logger
            )
        self.assertFalse(success)
        self.assertEqual(results, {image: app.WRITE_RESULT_UPDATED})
        self.assertEqual(error, "exiftool_write_failed")

    def test_write_fails_for_explicit_file_error_even_if_exiftool_exits_zero(self):
        image = os.path.abspath("a.jpg")

        def fake_run(command, logger, phase):
            args = Path(command[command.index("-@") + 1]).read_text(encoding="utf-8").splitlines()
            # Even a contradictory updated report must not hide an explicit failure.
            for flag in ("-efile", "-efile8"):
                Path(args[args.index(flag) + 1]).write_text(image + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(app, "run_exiftool", side_effect=fake_run):
            success, results, error = app.write_pass(
                [app.WriteTask(image, ["Liis"], app.REGION_INFO_KEEP)], False, self.logger
            )
        self.assertFalse(success)
        self.assertEqual(results, {image: app.WRITE_RESULT_FAILED})
        self.assertEqual(error, "exiftool_write_failed")

    def test_write_pass_handles_missing_exiftool_without_photos(self):
        image = os.path.abspath("a.jpg")
        with patch.object(app, "run_exiftool", return_value=None):
            self.assertEqual(
                app.write_pass([app.WriteTask(image, [], app.REGION_INFO_CLEAR)], False, self.logger),
                (False, {image: app.WRITE_RESULT_FAILED}, "exiftool_execution_failed"),
            )

    def test_cli_dry_run_writes_audit_but_never_calls_write_pass(self):
        with tempfile.TemporaryDirectory() as root:
            photo_dir = Path(root, "photos")
            photo_dir.mkdir()
            image = str(photo_dir / "a.jpg")
            sidecar = image + ".xmp"
            Path(image).write_bytes(b"untouched photo")
            Path(sidecar).write_bytes(b"untouched xmp")
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", str(photo_dir), "--mode", "replace", "--dry-run"]), \
                     patch.object(app, "read_pass", return_value={
                         image: {"Subject": "old"}, sidecar: {"Subject": "Liis"}
                     }), \
                     patch.object(app, "write_pass") as write_mock:
                    self.assertEqual(app.main(), 0)
                write_mock.assert_not_called()
                audit_path = next(Path(root, "logs").glob("face_embed_run_*.json"))
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(Path(image).read_bytes(), b"untouched photo")
            self.assertEqual(audit["summary"]["files_planned_for_write"], 1)
            self.assertEqual(audit["summary"]["files_updated"], 0)
            self.assertEqual(audit["files"][0]["write_result"], app.WRITE_RESULT_DRY_RUN)
            self.assertEqual(audit["run"]["mode"], "replace")

    def test_cli_read_failure_returns_error_and_audits_it(self):
        with tempfile.TemporaryDirectory() as root:
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=[("image.jpg", "image.jpg.xmp")]), \
                     patch.object(app, "read_pass", return_value=None), \
                     patch.object(app, "write_pass") as write_mock:
                    self.assertEqual(app.main(), 1)
                write_mock.assert_not_called()
                audit = json.loads(next(Path("logs").glob("*.json")).read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(audit["fatal_error"], "read_pass_failed")
            self.assertEqual(audit["summary"]["pairs_found"], 1)

    def test_cli_failed_write_returns_error_and_audits_each_file(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = str(Path(root, "a.jpg")), str(Path(root, "a.jpg.xmp"))
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root, "--mode", "replace"]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=[(image, sidecar)]), \
                     patch.object(app, "read_pass", return_value={image: {}, sidecar: {"Subject": "Liis"}}), \
                     patch.object(app, "write_pass", return_value=(
                         False, {image: app.WRITE_RESULT_FAILED}, "exiftool_write_failed"
                     )) as write_mock:
                    self.assertEqual(app.main(), 1)
                write_mock.assert_called_once()
                audit = json.loads(next(Path("logs").glob("*.json")).read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(audit["summary"]["files_failed"], 1)
            self.assertEqual(audit["files"][0]["write_result"], app.WRITE_RESULT_FAILED)
            self.assertEqual(audit["files"][0]["error"], "exiftool_write_failed")

    def test_cli_assigns_mixed_write_results_to_the_correct_images(self):
        with tempfile.TemporaryDirectory() as root:
            pairs = [(str(Path(root, f"{name}.jpg")), str(Path(root, f"{name}.jpg.xmp")))
                     for name in "abcd"]
            metadata_map = {
                pairs[0][0]: {}, pairs[0][1]: {"Subject": "Anna"},
                pairs[1][0]: {"Subject": "Same"}, pairs[1][1]: {"Subject": "Same"},
                pairs[2][0]: {}, pairs[2][1]: {"Subject": "Märt"},
                pairs[3][0]: {}, pairs[3][1]: {"Subject": "Liis"},
            }
            write_results = {
                pairs[0][0]: app.WRITE_RESULT_UPDATED,
                pairs[2][0]: app.WRITE_RESULT_FAILED,
                pairs[3][0]: app.WRITE_RESULT_UNCHANGED,
            }
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root, "--mode", "replace"]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=pairs), \
                     patch.object(app, "read_pass", return_value=metadata_map), \
                     patch.object(app, "write_pass", return_value=(
                         False, write_results, "exiftool_write_failed"
                     )) as write_mock:
                    self.assertEqual(app.main(), 1)
                self.assertEqual(
                    [task.image_path for task in write_mock.call_args.args[0]],
                    [pairs[index][0] for index in (0, 2, 3)],
                )
                audit = json.loads(next(Path("logs").glob("*.json")).read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(
                [item["image_path"] for item in audit["files"]],
                [image for image, _ in pairs],
            )
            self.assertEqual(
                [item["write_result"] for item in audit["files"]],
                [app.WRITE_RESULT_UPDATED, app.WRITE_RESULT_NOT_APPLICABLE,
                 app.WRITE_RESULT_FAILED, app.WRITE_RESULT_UNCHANGED],
            )
            self.assertEqual(
                [item["decision"] for item in audit["files"]],
                [app.DECISION_UPDATE, app.DECISION_UNCHANGED,
                 app.DECISION_UPDATE, app.DECISION_UPDATE],
            )
            self.assertEqual(
                [item["error"] for item in audit["files"]],
                [None, None, "exiftool_write_failed", None],
            )
            self.assertEqual(audit["summary"], {
                "pairs_found": 4, "files_evaluated": 4, "files_skipped": 0,
                "files_planned_for_write": 3, "files_updated": 1,
                "files_unchanged": 2, "files_failed": 1,
            })

    def test_cli_audit_write_error_returns_failure(self):
        with tempfile.TemporaryDirectory() as root:
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=[]), \
                     patch.object(app.json, "dump", side_effect=OSError("disk full")):
                    self.assertEqual(app.main(), 1)
            finally:
                os.chdir(old_cwd)

    def test_cli_defaults_to_fill_and_protects_existing_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = str(Path(root, "a.jpg")), str(Path(root, "a.jpg.xmp"))
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=[(image, sidecar)]), \
                     patch.object(app, "read_pass", return_value={
                         image: {"Subject": "Existing"}, sidecar: {"Subject": "New"}
                     }), \
                     patch.object(app, "write_pass") as write_mock:
                    self.assertEqual(app.main(), 0)
                write_mock.assert_not_called()
                audit = json.loads(next(Path("logs").glob("*.json")).read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(audit["run"]["mode"], app.MODE_FILL)
            self.assertEqual(audit["files"][0]["reasons"], [app.REASON_IMAGE_HAS_EXISTING_METADATA])

    def test_cli_forwards_keep_original_to_write_pass(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = str(Path(root, "a.jpg")), str(Path(root, "a.jpg.xmp"))
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", ["embed_faces.py", root, "--keep-original"]), \
                     patch.object(app, "find_image_xmp_pairs", return_value=[(image, sidecar)]), \
                     patch.object(app, "read_pass", return_value={
                         image: {}, sidecar: {"Subject": "Liis"}
                     }), \
                     patch.object(app, "write_pass", return_value=(
                         True, {image: app.WRITE_RESULT_UPDATED}, None
                     )) as write_mock:
                    self.assertEqual(app.main(), 0)
                self.assertTrue(write_mock.call_args.kwargs["keep_original"])
                self.assertEqual([task.image_path for task in write_mock.call_args.args[0]], [image])
                audit = json.loads(next(Path("logs").glob("*.json")).read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
            self.assertTrue(audit["run"]["keep_original"])
            self.assertEqual(audit["files"][0]["write_result"], app.WRITE_RESULT_UPDATED)

    @unittest.skipUnless(shutil.which(app.EXIFTOOL_EXECUTABLE), "ExifTool is not on PATH")
    def test_real_exiftool_round_trip_on_synthetic_image(self):
        with tempfile.TemporaryDirectory() as root:
            image, sidecar = create_synthetic_image_and_sidecar(Path(root))
            sidecar_digest = hashlib.sha256(Path(sidecar).read_bytes()).digest()
            pairs = [(image, sidecar)]
            before = app.read_pass(pairs, self.logger)
            self.assertIsNotNone(before)
            self.assertEqual(app.normalize_subjects(before[sidecar].get("Subject")), ["Synthetic Person"])
            regions = app.normalize_region_info(before[sidecar].get("RegionInfo"), sidecar)
            self.assertEqual(len(regions.get("RegionList", [])), 1)
            self.assertEqual(regions["RegionList"][0]["Name"], "Synthetic Person")
            tasks, _ = app.prepare_write_tasks(pairs, before, app.MODE_REPLACE, self.logger)
            self.assertEqual(len(tasks), 1)
            success, results, error = app.write_pass(tasks, keep_original=False, logger=self.logger)
            self.assertTrue(success, (results, error))
            self.assertEqual(results[image], app.WRITE_RESULT_UPDATED)
            after = app.read_pass(pairs, self.logger)
            self.assertIsNotNone(after)
            self.assertEqual(app.normalize_subjects(after[image].get("Subject")), tasks[0].subjects)
            actual = app.normalize_region_info(after[image].get("RegionInfo"), image)
            expected_info = tasks[0].region_info
            expected = expected_info["RegionList"]
            self.assertEqual(
                [app.region_real_name(r) for r in actual.get("RegionList", [])],
                [app.region_real_name(r) for r in expected],
            )
            self.assertEqual(len(actual.get("RegionList", [])), len(expected))
            for written, planned in zip(actual["RegionList"], expected):
                self.assertEqual(written.get("Type"), planned.get("Type"))
                self.assertEqual(written["Area"].get("Unit"), planned["Area"].get("Unit"))
                for key in ("X", "Y", "W", "H"):
                    self.assertAlmostEqual(
                        float(written["Area"][key]), float(planned["Area"][key]), delta=0.0001
                    )
            if "AppliedToDimensions" in expected_info:
                for key in ("W", "H"):
                    self.assertEqual(
                        float(actual["AppliedToDimensions"][key]),
                        float(expected_info["AppliedToDimensions"][key]),
                    )
                self.assertEqual(
                    actual["AppliedToDimensions"].get("Unit"),
                    expected_info["AppliedToDimensions"].get("Unit"),
                )
            self.assertEqual(hashlib.sha256(Path(sidecar).read_bytes()).digest(), sidecar_digest)
            repeat_tasks, repeat_evaluations = app.prepare_write_tasks(
                pairs, after, app.MODE_REPLACE, self.logger
            )
            self.assertEqual(repeat_tasks, [])
            self.assertEqual(repeat_evaluations[0].decision, app.DECISION_UNCHANGED)

    def test_audit_counts_skips_unchanged_updates_and_failures(self):
        def evaluation(decision, result):
            return app.EvaluatedFile("a.jpg", "a.jpg.xmp", [], {}, {}, decision, result)

        evaluations = [
            evaluation(app.DECISION_SKIP, app.WRITE_RESULT_NOT_APPLICABLE),
            evaluation(app.DECISION_UNCHANGED, app.WRITE_RESULT_NOT_APPLICABLE),
            evaluation(app.DECISION_UPDATE, app.WRITE_RESULT_UPDATED),
            evaluation(app.DECISION_UPDATE, app.WRITE_RESULT_UNCHANGED),
            evaluation(app.DECISION_UPDATE, app.WRITE_RESULT_FAILED),
        ]
        audit = app.build_run_audit("now", "/photos", "merge", False, False, 5, evaluations)
        self.assertEqual(audit["summary"], {
            "pairs_found": 5, "files_evaluated": 5, "files_skipped": 1,
            "files_planned_for_write": 3, "files_updated": 1,
            "files_unchanged": 2, "files_failed": 1,
        })


if __name__ == "__main__":
    unittest.main()
