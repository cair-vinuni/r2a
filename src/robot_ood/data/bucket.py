"""Read a trained artifact tree out of the project's Hugging Face bucket.

A bucket is not a model or dataset repo: snapshot_download returns 404 on one, and only
huggingface_hub >= 1 can list it. The run-time envs are pinned to 0.36, so the reader lives in its
own venv under .local/ and every download runs as a subprocess of that interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from robot_ood.core.paths import repo_root

__all__ = ["DEFAULT_BUCKET", "PUBLISHED", "download", "reader_python", "upload"]

DEFAULT_BUCKET = "twanghcmut/robot-ood-artifact"
READER_DIR = ".local/hf-bucket"
READER_SPEC = "huggingface_hub[hf_xet]>=1.28"

_DOWNLOAD = """
import json, sys
from pathlib import Path
from huggingface_hub import HfApi

bucket, source, dest = sys.argv[1], sys.argv[2].strip("/") + "/", Path(sys.argv[3])
api = HfApi()
files = [f for f in api.list_bucket_tree(bucket) if f.path.startswith(source)]
if not files:
    raise SystemExit(f"nothing under {source} in {bucket}")
api.download_bucket_files(bucket, [(f, dest / f.path[len(source):]) for f in files])
print(json.dumps({"files": len(files), "bytes": sum(f.size for f in files)}))
"""

# The tree a suite publishes: the artifact directory and the constants file beside it, and nothing
# else under outputs/<suite>/.
PUBLISHED: Final[tuple[str, ...]] = ("artifacts/*", "constants.json")

_UPLOAD = """
import json, sys
from huggingface_hub import HfApi

bucket, source, dest, include = sys.argv[1], sys.argv[2], sys.argv[3].strip("/"), sys.argv[4:]
plan = HfApi().sync_bucket(source, f"hf://buckets/{bucket}/{dest}", include=include, delete=True,
                           quiet=True)
done = plan.summary()
print(json.dumps({"files": done["uploads"], "deleted": done["deletes"],
                  "bytes": done["total_size"]}))
"""


def reader_python() -> Path:
    """The bucket reader's interpreter, building its venv on first use.

    Returns:
        The path to .local/hf-bucket/bin/python.
    """
    reader = repo_root() / READER_DIR / "bin" / "python"
    if _reader_ok(reader):
        return reader
    # A half-built or stale venv (dead interpreter, no huggingface_hub) is rebuilt, not reused.
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(repo_root() / READER_DIR)],
                   check=True)
    subprocess.run([str(reader), "-m", "pip", "install", "-q", READER_SPEC], check=True)
    if not _reader_ok(reader):
        raise RuntimeError(f"bucket reader at {reader} cannot import huggingface_hub")
    return reader


def _reader_ok(reader: Path) -> bool:
    """Whether the reader runs and has a huggingface_hub that can list buckets."""
    if not reader.exists():
        return False
    probe = "from huggingface_hub import HfApi; HfApi.list_bucket_tree"
    return subprocess.run([str(reader), "-c", probe], capture_output=True).returncode == 0


def download(source: str, dest: Path, bucket: str = DEFAULT_BUCKET) -> dict:
    """Copy one bucket subdirectory into a local directory.

    Args:
        source: The subdirectory inside the bucket.
        dest: Where its contents land; created if absent.
        bucket: The bucket id.

    Returns:
        {"files": n, "bytes": total} for what was downloaded.

    Raises:
        RuntimeError: The bucket holds nothing under that subdirectory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(reader_python()), "-c", _DOWNLOAD, bucket, source, str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bucket read failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def upload(source: Path, dest: str, bucket: str = DEFAULT_BUCKET) -> dict:
    """Publish PUBLISHED out of a suite output directory into one bucket subdirectory.

    Args:
        source: The suite output directory; PUBLISHED selects what leaves it.
        dest: The subdirectory inside the bucket, created if absent.
        bucket: The bucket id.

    Returns:
        {"files": n, "deleted": n, "bytes": total} for what the sync did.

    Raises:
        RuntimeError: The bucket write failed.
    """
    proc = subprocess.run(
        [str(reader_python()), "-c", _UPLOAD, bucket, str(source), dest, *PUBLISHED],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bucket write failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])
