#!/usr/bin/env python3
"""Download, verify and extract models listed in models/catalog.tsv. Safe to re-run."""

import argparse
import csv
import hashlib
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "models" / "catalog.tsv"


def fetch_archive(row: dict, output_directory: Path) -> str:
    target_directory = output_directory / row["zoo_directory"] / row["archive"]
    done_marker = target_directory / ".sha256"
    if done_marker.is_file():
        return f"{done_marker.read_text().strip()}\t{row['url']}"

    target_directory.mkdir(parents=True, exist_ok=True)
    archive_path = target_directory / "archive.partial"
    digest = hashlib.sha256()
    with urllib.request.urlopen(row["url"], timeout=120) as response, archive_path.open("wb") as output:
        for block in iter(lambda: response.read(CHUNK_BYTES), b""):
            digest.update(block)
            output.write(block)
    expected_sha256 = row["sha256"]
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        archive_path.unlink()
        raise ValueError(f"sha256 {digest.hexdigest()} != pinned {expected_sha256}")
    with tarfile.open(archive_path) as archive:
        # filter="data" rejects absolute paths, traversal and special files.
        archive.extractall(target_directory, filter="data")
    archive_path.unlink()
    done_marker.write_text(digest.hexdigest() + "\n")
    return f"{digest.hexdigest()}\t{row['url']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()

    with arguments.catalog.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if arguments.include_optional or row["tier"] == "default"
        ]

    # Smallest first: on a slow link, many useful models arrive before the huge ones.
    rows.sort(key=lambda row: int(row["size_mb"]))

    def attempt(row: dict):
        try:
            return row, fetch_archive(row, arguments.output_directory), None
        except Exception as error:  # One bad archive must not abort a multi-hour fetch.
            return row, None, error

    failure_count = 0
    checksum_lines = []
    with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        for row, checksum_line, error in pool.map(attempt, rows):
            if error is not None:
                failure_count += 1
                print(f"FAILED {row['zoo_directory']}/{row['archive']}: {error}", flush=True)
            else:
                checksum_lines.append(checksum_line)
                print(f"OK {row['zoo_directory']}/{row['archive']}", flush=True)
    checksums_path = arguments.output_directory / "checksums.tsv"
    previous_lines = checksums_path.read_text().splitlines() if checksums_path.is_file() else []
    checksums_path.write_text("\n".join(sorted({*previous_lines, *checksum_lines})) + "\n")
    print(f"Finished: {len(checksum_lines)} ok, {failure_count} failed", flush=True)
    return 1 if failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
