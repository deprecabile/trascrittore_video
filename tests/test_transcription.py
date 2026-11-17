import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from google.genai import interactions


ROOT_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trascrittore_gemini", ROOT_DIR / "src" / "trascrittore_gemini.py"
)
transcriber = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}), patch("google.genai.Client"):
    spec.loader.exec_module(transcriber)


def remote_file(state="ACTIVE"):
    return SimpleNamespace(
        name="files/test-audio", state=SimpleNamespace(name=state),
        uri="https://generativelanguage.googleapis.com/v1beta/files/test-audio",
        mime_type="audio/mpeg",
    )


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.files.upload.return_value = remote_file()
        self.client.interactions.create.return_value = SimpleNamespace(
            status="completed", output_text="  Trascrizione completa.  "
        )
        self.enterContext(patch.object(transcriber, "client", self.client))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def test_smart_transcription_is_stateless_and_deletes_upload(self):
        self.assertEqual(transcriber.trascrivi_blocco("audio.mp3"), "Trascrizione completa.")
        self.client.interactions.create.assert_called_once_with(
            model="gemini-3.5-transcribe",
            input=[{"type": "audio", "uri": remote_file().uri, "mime_type": "audio/mpeg"}],
            generation_config={"transcription_config": {"mode": "smart"}},
            store=False,
        )
        self.client.files.delete.assert_called_once_with(name="files/test-audio")
        self.client.models.generate_content.assert_not_called()
        # Verifica anche lo schema reale dell'SDK, non soltanto il mock.
        request = interactions.CreateModelInteraction.model_validate(
            self.client.interactions.create.call_args.kwargs
        )
        self.assertEqual(request.generation_config.transcription_config.mode, "smart")

    def test_cleanup_failure_preserves_original_api_error(self):
        original_error = RuntimeError("503 UNAVAILABLE")
        self.client.interactions.create.side_effect = original_error
        self.client.files.delete.side_effect = RuntimeError("Cleanup unavailable")
        with self.assertRaises(RuntimeError) as caught:
            transcriber.trascrivi_blocco("audio.mp3")
        self.assertIs(caught.exception, original_error)
        self.client.files.delete.assert_called_once()

    def test_incomplete_or_empty_response_is_rejected(self):
        for status, text in [("failed", "Parziale"), ("completed", None), ("completed", "  ")]:
            with self.subTest(status=status, text=text):
                self.client.interactions.create.return_value = SimpleNamespace(
                    status=status, output_text=text
                )
                with self.assertRaises(RuntimeError):
                    transcriber.trascrivi_blocco("audio.mp3")
        self.assertEqual(self.client.files.delete.call_count, 3)

    def test_waits_for_processing_before_transcribing(self):
        self.client.files.upload.return_value = remote_file("PROCESSING")
        self.client.files.get.return_value = remote_file()
        with patch.object(transcriber.time, "sleep"):
            transcriber.trascrivi_blocco("audio.mp3")
        self.client.files.get.assert_called_once_with(name="files/test-audio")
        self.client.interactions.create.assert_called_once()

    def test_processing_failure_and_timeout_delete_upload(self):
        for state in ["FAILED", "PROCESSING"]:
            with self.subTest(state=state):
                self.client.files.upload.return_value = remote_file(state)
                with patch.object(transcriber.time, "monotonic", side_effect=[0, 301]):
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        transcriber.trascrivi_blocco("audio.mp3")
        self.client.interactions.create.assert_not_called()
        self.assertEqual(self.client.files.delete.call_count, 2)

    def test_chunks_are_joined_in_order_and_temporary_directory_is_removed(self):
        directories = []
        def prepare(audio, directory):
            directories.append(Path(directory))
            return [Path(directory) / "blocco_0000.mp3", Path(directory) / "blocco_0001.mp3"]
        with patch.object(transcriber, "prepara_blocchi_audio", side_effect=prepare), \
                patch.object(transcriber, "trascrivi_blocco", side_effect=["Prima parte", "Seconda parte"]):
            self.assertEqual(transcriber.trascrivi_con_gemini("audio.mp3"), "Prima parte\n\nSeconda parte")
        self.assertFalse(directories[0].exists())


class AudioChunkTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT_DIR / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(dir=cache)))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def test_short_audio_is_used_without_reencoding(self):
        result = subprocess.CompletedProcess([], 0, stdout="773.2\n")
        with patch.object(transcriber.subprocess, "run", return_value=result) as run:
            blocks = transcriber.prepara_blocchi_audio("audio.mp3", self.directory)
        self.assertEqual(blocks, [Path("audio.mp3").resolve()])
        self.assertEqual(run.call_count, 1)

    def test_long_audio_is_split_and_sorted(self):
        def run(command, **kwargs):
            if command[0] == "ffprobe":
                return subprocess.CompletedProcess(command, 0, stdout="3700\n")
            for index in [2, 0, 1]:
                (self.directory / f"blocco_{index:04d}.mp3").write_bytes(b"audio")
            self.assertEqual(command[command.index("-segment_time") + 1], "1800")
            self.assertEqual(command[command.index("-c:a") + 1], "copy")
            return subprocess.CompletedProcess(command, 0)
        with patch.object(transcriber.subprocess, "run", side_effect=run):
            blocks = transcriber.prepara_blocchi_audio("audio.mp3", self.directory)
        self.assertEqual([block.name for block in blocks], [f"blocco_{i:04d}.mp3" for i in range(3)])

    def test_invalid_duration_does_not_upload_or_split(self):
        for duration in ["nan", "inf", "0", "-1", "N/A"]:
            with self.subTest(duration=duration):
                result = subprocess.CompletedProcess([], 0, stdout=duration)
                with patch.object(transcriber.subprocess, "run", return_value=result) as run:
                    with self.assertRaises(ValueError):
                        transcriber.prepara_blocchi_audio("audio.mp3", self.directory)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
