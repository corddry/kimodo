
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spark-specific post-sync repairs for the shared Kimodo install.

The DGX Spark runs Linux/aarch64. A few binary packages need local metadata or
rpath repairs after a fresh `uv sync`:

* The vendored Linux/aarch64 embreex wheel should resolve Embree/TBB from the
  shared runtime prefix.
* NVIDIA's cusparselt wheel is tagged `manylinux2014_sbsa`, which works on the
  Spark but is not considered compatible by uv's aarch64 tag matcher. Rewriting
  the installed WHEEL tag prevents uv from uninstalling/reinstalling it on every
  sync.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

EMBREE_RUNTIME = Path("/srv/shared/deps/embree-4.3-arm64/lib")
CUSPARSELT_DIST_INFO = "nvidia_cusparselt_cu13-0.8.1.dist-info"
CUSPARSELT_UV_COMPAT_TAG = "Tag: py3-none-manylinux_2_17_aarch64"
CUSPARSELT_UPSTREAM_TAG = "Tag: py3-none-manylinux2014_sbsa"


def _site_packages() -> Path:
    return Path(__file__).resolve().parents[2]


def _patch_rpath(path: Path, rpath: Path) -> bool:
    if not path.exists():
        return False
    current = subprocess.run(
        ["patchelf", "--print-rpath", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    if str(rpath) == current:
        return False
    subprocess.run(["patchelf", "--set-rpath", str(rpath), str(path)], check=True)
    return True


def patch_embreex_rpaths(site_packages: Path | None = None) -> int:
    site_packages = site_packages or _site_packages()
    changed = 0
    embreex_dir = site_packages / "embreex"
    if embreex_dir.exists():
        for extension in embreex_dir.glob("*.so"):
            changed += int(_patch_rpath(extension, EMBREE_RUNTIME))

    if EMBREE_RUNTIME.exists():
        for embree_lib in EMBREE_RUNTIME.glob("libembree4.so*"):
            if embree_lib.is_file() or embree_lib.is_symlink():
                changed += int(_patch_rpath(embree_lib, EMBREE_RUNTIME))
    return changed


def patch_cusparselt_wheel_tag(site_packages: Path | None = None) -> bool:
    site_packages = site_packages or _site_packages()
    wheel = site_packages / CUSPARSELT_DIST_INFO / "WHEEL"
    if not wheel.exists():
        return False
    text = wheel.read_text()
    if CUSPARSELT_UV_COMPAT_TAG in text:
        return False
    if CUSPARSELT_UPSTREAM_TAG not in text:
        return False
    wheel.write_text(text.replace(CUSPARSELT_UPSTREAM_TAG, CUSPARSELT_UV_COMPAT_TAG))
    return True


def main() -> None:
    rpaths = patch_embreex_rpaths()
    cusparselt = patch_cusparselt_wheel_tag()
    print(f"spark post-sync repairs: embreex_rpaths_changed={rpaths} cusparselt_tag_changed={cusparselt}")


if __name__ == "__main__":
    main()
