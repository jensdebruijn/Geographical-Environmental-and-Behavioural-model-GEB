import subprocess
from geb.cli import multi_level_merge


def test_multi_level_merge():
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 2, "c": {"d": 4, "f": 5}}

    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 2, "b": 2, "c": {"d": 4, "e": 4, "f": 5}}


def test_cli():
    for cmd in ("run", "build", "alter", "update", "calibrate"):
        result = subprocess.run(["geb", cmd, "--help"], capture_output=True, text=True)

        # Check the exit code
        assert result.returncode == 0
