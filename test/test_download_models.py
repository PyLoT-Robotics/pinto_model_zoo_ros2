import hashlib
import importlib.util
import io
import re
import tarfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
specification = importlib.util.spec_from_file_location("download_models", SCRIPT_PATH)
download_models = importlib.util.module_from_spec(specification)
specification.loader.exec_module(download_models)


def make_archive(directory: Path, members: dict[str, bytes]) -> Path:
    archive_path = directory / "resources.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return archive_path


def test_archive_table_has_unique_names_and_valid_hashes():
    names = [name for spec in download_models.ARCHIVES for name in spec.files]
    assert len(names) == len(set(names))
    for spec in download_models.ARCHIVES:
        for sha256 in spec.files.values():
            assert re.fullmatch(r"[0-9a-f]{64}", sha256)


def test_extracts_only_listed_files_and_verifies_hash(tmp_path):
    content = b"weights"
    archive_path = make_archive(tmp_path, {"model.onnx": content, "../evil.onnx": b"x"})
    output_directory = tmp_path / "out"
    output_directory.mkdir()
    expected = {"model.onnx": hashlib.sha256(content).hexdigest()}

    download_models.extract_verified_models(archive_path, expected, output_directory)

    assert [path.name for path in output_directory.iterdir()] == ["model.onnx"]


def test_hash_mismatch_raises_and_leaves_no_file(tmp_path):
    archive_path = make_archive(tmp_path, {"model.onnx": b"weights"})
    output_directory = tmp_path / "out"
    output_directory.mkdir()

    with pytest.raises(ValueError, match="sha256"):
        download_models.extract_verified_models(
            archive_path, {"model.onnx": "0" * 64}, output_directory
        )

    assert list(output_directory.iterdir()) == []
