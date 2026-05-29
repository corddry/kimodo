from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARK_MARKER = "platform_machine == 'aarch64' and sys_platform == 'linux' and python_version == '3.10'"
EMBREEX_WHEEL = "wheels/embreex-4.4.0-cp310-cp310-linux_aarch64.whl"
EMBREE_RUNTIME = b"/srv/shared/deps/embree-4.3-arm64/lib"


def test_spark_aarch64_embreex_wheel_is_managed_by_uv_sources():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "embreex = [" in pyproject
    assert f'path = "{EMBREEX_WHEEL}"' in pyproject
    assert f'marker = "{SPARK_MARKER}"' in pyproject


def test_spark_extra_keeps_acceleration_and_demo_dependencies_managed():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "spark = [" in pyproject
    for dependency in (
        '"embreex==4.4.0"',
        '"warp-lang"',
        '"cholespy"',
        '"viser @ git+https://github.com/nv-tlabs/kimodo-viser.git"',
        '"py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git"',
    ):
        assert dependency in pyproject


def test_vendored_spark_embreex_wheel_has_runtime_rpath():
    wheel = ROOT / EMBREEX_WHEEL
    assert wheel.is_file()

    with zipfile.ZipFile(wheel) as zf:
        extension_names = [name for name in zf.namelist() if name.startswith("embreex/") and name.endswith(".so")]
        assert extension_names
        for name in extension_names:
            data = zf.read(name)
            assert EMBREE_RUNTIME in data, name


if __name__ == "__main__":
    for test in (
        test_spark_aarch64_embreex_wheel_is_managed_by_uv_sources,
        test_spark_extra_keeps_acceleration_and_demo_dependencies_managed,
        test_vendored_spark_embreex_wheel_has_runtime_rpath,
    ):
        test()
        print(f"{test.__name__}: OK")
