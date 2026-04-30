"""Tests for utility helpers."""

import os

import pytest

from kimi_acp_bridge.utils import build_controlled_env, validate_work_dir


class TestValidateWorkDir:
    """Test work directory validation."""

    def test_rejects_relative_path(self):
        with pytest.raises(ValueError, match="absolute"):
            validate_work_dir("relative/path")

    def test_rejects_nonexistent_path(self):
        with pytest.raises(ValueError, match="does not exist"):
            validate_work_dir("/nonexistent_path_12345")

    def test_accepts_valid_absolute_directory(self, tmp_path):
        result = validate_work_dir(str(tmp_path))
        assert os.path.samefile(result, str(tmp_path))

    def test_rejects_file_path(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            validate_work_dir(str(f))


class TestBuildControlledEnv:
    """Test controlled environment building."""

    def test_includes_allowed_keys(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("SECRET_API_KEY", "should-not-appear")

        env = build_controlled_env()
        assert "PATH" in env
        assert "HOME" in env
        assert "SECRET_API_KEY" not in env

    def test_extra_vars_merged(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        env = build_controlled_env(extra={"EXTRA": "value"})
        assert "EXTRA" in env
        assert env["EXTRA"] == "value"
