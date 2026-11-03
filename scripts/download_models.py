#!/usr/bin/env python3
"""Fetch PINTO model zoo archives and extract the pinned ONNX models."""

import argparse
import hashlib
import socket
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ZOO_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/"
DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ArchiveSpec:
    url: str
    files: dict[str, str]  # file name inside the archive -> pinned sha256


ARCHIVES = (
    ArchiveSpec(
        url=ZOO_URL + "488_DEIMv2-Wholebody49/resources.tar.gz",
        files={
            "deimv2_hgnetv2_n_wholebody49_ins_s08_maskhead64x2_center_1240query.onnx":
                "b02cae5f12ebaa7f57be35e5bd1b83e58273472c69b2eb87696954adae83a95c",
            "deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx":
                "a8c46528408e6bca47a119e73f3ccfd4a7d3debe0c083fed29ed87789ce3eeb5",
            "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx":
                "707f14deb30fcd0c0d437b7a87f4d246f734ed808968bf731f7da159fdfd5871",
        },
    ),
    ArchiveSpec(
        url=ZOO_URL + "502_PersonViT/resources_ain.tar.gz",
        files={
            "osnet_ain_x1_0_p_unified_aug_n.onnx":
                "0235656a0db94265491d0df90ebcbc3543c1dd61937a38b82eeebd31c76cb220",
        },
    ),
    ArchiveSpec(
        url=ZOO_URL + "483_LVFace/resources.tar.gz",
        files={
            "LVFace-T_Glint360K.onnx":
                "6635071a23a06aca91ddfd87e3874c7ffd42fc41a39a50f833e943c88a59df91",
        },
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_archive_present(spec: ArchiveSpec, output_directory: Path) -> bool:
    return all(
        (output_directory / name).is_file()
        and sha256_of(output_directory / name) == sha
        for name, sha in spec.files.items()
    )


def extract_verified_models(
    archive_path: Path, expected_files: dict[str, str], output_directory: Path
) -> None:
    """Extract only pinned names, so archive contents never choose a write path."""
    with tarfile.open(archive_path) as archive:
        for file_name, expected_sha256 in expected_files.items():
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
                    "Upstream may have changed the weights; review, then update ARCHIVES."
                )
            temporary_path.replace(output_directory / file_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_SECONDS)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    for spec in ARCHIVES:
        if is_archive_present(spec, arguments.output_directory):
            continue
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "resources.tar.gz"
            print(f"Downloading {spec.url}")
            urllib.request.urlretrieve(spec.url, archive_path)
            extract_verified_models(archive_path, spec.files, arguments.output_directory)
    model_count = sum(len(spec.files) for spec in ARCHIVES)
    print(f"Verified {model_count} models in {arguments.output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
