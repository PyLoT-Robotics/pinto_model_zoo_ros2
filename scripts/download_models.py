#!/usr/bin/env python3
"""Fetch PINTO's DEIMv2-Wholebody49 archive and extract the pinned ONNX models."""

import argparse
import hashlib
import socket
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = (
    "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/"
    "488_DEIMv2-Wholebody49/resources.tar.gz"
)
DOWNLOAD_TIMEOUT_SECONDS = 120
MODEL_SHA256 = {
    "deimv2_hgnetv2_n_wholebody49_ins_s08_maskhead64x2_center_1240query.onnx":
        "b02cae5f12ebaa7f57be35e5bd1b83e58273472c69b2eb87696954adae83a95c",
    "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx":
        "a8c46528408e6bca47a119e73f3ccfd4a7d3debe0c083fed29ed87789ce3eeb5",
    "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx":
        "707f14deb30fcd0c0d437b7a87f4d246f734ed808968bf731f7da159fdfd5871",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_verified_models(archive_path: Path, output_directory: Path) -> None:
    """Extract only pinned names, so archive contents never choose a write path."""
    with tarfile.open(archive_path) as archive:
        for file_name, expected_sha256 in MODEL_SHA256.items():
            member = archive.extractfile(file_name)
            if member is None:
                raise ValueError(f"{file_name} is not a regular file in the archive")
            temporary_path = output_directory / f"{file_name}.partial"
            with member, temporary_path.open("wb") as output:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    output.write(block)
            actual_sha256 = sha256_of(temporary_path)
            if actual_sha256 != expected_sha256:
                temporary_path.unlink()
                raise ValueError(
                    f"{file_name}: sha256 {actual_sha256} != pinned {expected_sha256}. "
                    "Upstream may have changed the weights; review, then update MODEL_SHA256."
                )
            temporary_path.replace(output_directory / file_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_SECONDS)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    is_complete = all(
        (arguments.output_directory / name).is_file()
        and sha256_of(arguments.output_directory / name) == sha
        for name, sha in MODEL_SHA256.items()
    )
    if is_complete:
        print("Models already present and verified.")
        return 0

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "resources.tar.gz"
        print(f"Downloading {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
        extract_verified_models(archive_path, arguments.output_directory)
    print(f"Verified {len(MODEL_SHA256)} models in {arguments.output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
