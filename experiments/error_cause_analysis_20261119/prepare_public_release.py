"""Build deterministic compressed copies and SHA256 manifest for public outputs."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
COMPRESS = (
    "local_stratified_14px_diagnostics.csv",
    "per_image_local_diagnostics.csv",
)
EXCLUDE = {
    "analysis/local_stratified_14px_diagnostics.csv",
    "analysis/per_image_local_diagnostics.csv",
    "SHA256SUMS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for filename in COMPRESS:
        source = ANALYSIS / filename
        target = source.with_name(source.name + ".gz")
        with source.open("rb") as source_stream, target.open("wb") as target_stream:
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, fileobj=target_stream, mtime=0
            ) as compressed:
                compressed.write(source_stream.read())
        if gzip.decompress(target.read_bytes()) != source.read_bytes():
            raise RuntimeError(f"gzip round-trip mismatch: {target}")

    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "figures" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.parts
        and path.relative_to(ROOT).as_posix() not in EXCLUDE
    )
    manifest = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files
    )
    (ROOT / "SHA256SUMS").write_text(manifest, encoding="utf-8", newline="\n")
    for line in manifest.splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch after writing manifest: {relative}")
    print(f"PASS: {len(files)} public files; gzip round-trip verified")


if __name__ == "__main__":
    main()
