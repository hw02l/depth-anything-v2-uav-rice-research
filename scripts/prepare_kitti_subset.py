from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import zipfile

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEFAULT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"


class HTTPRangeFile(io.RawIOBase):
    """A seekable file-like object backed by HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, verify=False, timeout=60)
        head.raise_for_status()
        self.length = int(head.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self.position + offset
        elif whence == 2:
            target = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self.position = max(0, min(self.length, target))
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        end = self.length - 1 if size is None or size < 0 else min(self.length - 1, self.position + size - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
            verify=False,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.content
        if response.status_code == 200 and len(payload) == self.length:
            payload = payload[self.position : end + 1]
        self.position += len(payload)
        return payload


def matching_ground_truth(image_member: str, names: set[str]) -> str | None:
    candidate = image_member.replace("/image/", "/groundtruth_depth/")
    candidate = candidate.replace("_sync_image_", "_sync_groundtruth_depth_", 1)
    if candidate in names:
        return candidate
    basename = Path(image_member).name
    candidates = [name for name in names if Path(name).name == basename]
    if candidates:
        return sorted(candidates)[0]
    # The official archive occasionally changes only the directory/prefix.
    token = re.sub(r"_sync_image_", "_sync_", basename)
    candidates = [name for name in names if token in re.sub(r"_sync_groundtruth_depth_", "_sync_", Path(name).name)]
    return sorted(candidates)[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small official KITTI validation subset without downloading the full ZIP")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--outdir", default="data/kitti_validation_50")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    outdir = Path(args.outdir)
    image_dir = outdir / "val_selection_cropped" / "image"
    gt_dir = outdir / "val_selection_cropped" / "groundtruth_depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[kitti] opening remote archive: {args.url}")
    with zipfile.ZipFile(HTTPRangeFile(args.url)) as archive:
        names = set(archive.namelist())
        images = sorted(
            name for name in names
            if "/val_selection_cropped/image/" in name and name.lower().endswith(".png")
        )
        if not images:
            raise RuntimeError("No val_selection_cropped/image PNGs were found in the archive")
        selected_indices = [round(index * (len(images) - 1) / max(1, args.count - 1)) for index in range(min(args.count, len(images)))]
        selected = [images[index] for index in selected_indices]
        manifest = []
        for image_member in selected:
            gt_member = matching_ground_truth(image_member, names)
            if gt_member is None:
                raise FileNotFoundError(f"Could not match ground truth for {image_member}")
            image_path = image_dir / Path(image_member).name
            gt_path = gt_dir / Path(gt_member).name
            image_path.write_bytes(archive.read(image_member))
            gt_path.write_bytes(archive.read(gt_member))
            manifest.append({"image_member": image_member, "groundtruth_member": gt_member, "image": image_path.name, "groundtruth": gt_path.name})
            print(f"[kitti] extracted {image_path.name}")

    (outdir / "manifest.json").write_text(
        json.dumps({"source_url": args.url, "archive_images": len(images), "selected": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[kitti] wrote {len(manifest)} image/GT pairs to {outdir}")


if __name__ == "__main__":
    main()
