from __future__ import annotations

from pathlib import Path

import pytest

import furnace.config
from furnace.config import ToolPaths, load_config

_MANDATORY = (
    "ffmpeg",
    "ffprobe",
    "mkvmerge",
    "mkvpropedit",
    "mkclean",
    "eac3to",
    "qaac64",
    "mpv",
    "makemkvcon",
    "nvencc",
)


def _write_config(
    tmp_path: Path,
    *,
    tools_override: dict[str, str] | None = None,
    omit_keys: tuple[str, ...] = (),
) -> Path:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    for name in _MANDATORY:
        if name in omit_keys:
            continue
        p = tool_dir / f"{name}.exe"
        p.touch()
        paths[name] = str(p)
    if tools_override:
        paths.update(tools_override)
    lines = ["[tools]"]
    for k, v in paths.items():
        lines.append(f'{k} = "{v}"')
    config = tmp_path / "furnace.toml"
    config.write_text("\n".join(lines))
    return config


class TestExplicitPath:
    def test_valid_config_returns_tool_paths(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        tp = load_config(config)
        assert isinstance(tp, ToolPaths)
        for name in _MANDATORY:
            assert getattr(tp, name) == tmp_path / "tools" / f"{name}.exe"

    def test_missing_explicit_path_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        with pytest.raises(FileNotFoundError, match="explicit path"):
            load_config(missing)


class TestMissingMandatoryKey:
    @pytest.mark.parametrize("omit", _MANDATORY)
    def test_missing_key_raises(self, tmp_path: Path, omit: str) -> None:
        config = _write_config(tmp_path, omit_keys=(omit,))
        with pytest.raises(KeyError, match=f"\\[tools\\]\\.{omit}"):
            load_config(config)


class TestToolPathMissing:
    def test_nonexistent_tool_path_raises(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "tools" / "no_such_binary.exe")
        config = _write_config(tmp_path, tools_override={"ffmpeg": bogus})
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            load_config(config)


class TestCwdSearch:
    def test_finds_config_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("APPDATA", raising=False)
        tp = load_config()
        assert isinstance(tp, ToolPaths)
        assert tp.ffmpeg == tmp_path / "tools" / "ffmpeg.exe"


class TestNotFoundAnywhere:
    def test_no_config_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("APPDATA", raising=False)
        fake_pkg = tmp_path / "fake_pkg"
        fake_pkg.mkdir()
        monkeypatch.setattr(furnace.config, "__file__", str(fake_pkg / "config.py"))
        with pytest.raises(FileNotFoundError, match=r"No furnace\.toml config found"):
            load_config()


class TestDoviTool:
    def test_present(self, tmp_path: Path) -> None:
        dovi = tmp_path / "tools" / "dovi_tool.exe"
        config = _write_config(tmp_path, tools_override={"dovi_tool": str(dovi)})
        dovi.touch()
        tp = load_config(config)
        assert tp.dovi_tool == dovi

    def test_absent(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        tp = load_config(config)
        assert tp.dovi_tool is None

    def test_dovi_tool_path_missing_raises(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "tools" / "no_dovi.exe")
        config = _write_config(tmp_path, tools_override={"dovi_tool": bogus})
        with pytest.raises(FileNotFoundError, match="dovi_tool"):
            load_config(config)


class TestHdr10PlusTool:
    def test_present(self, tmp_path: Path) -> None:
        hdr10plus = tmp_path / "tools" / "hdr10plus_tool.exe"
        config = _write_config(tmp_path, tools_override={"hdr10plus_tool": str(hdr10plus)})
        hdr10plus.touch()
        tp = load_config(config)
        assert tp.hdr10plus_tool == hdr10plus

    def test_absent(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        tp = load_config(config)
        assert tp.hdr10plus_tool is None

    def test_hdr10plus_tool_path_missing_raises(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "tools" / "no_hdr10plus.exe")
        config = _write_config(tmp_path, tools_override={"hdr10plus_tool": bogus})
        with pytest.raises(FileNotFoundError, match="hdr10plus_tool"):
            load_config(config)


class TestVapourSynthPlugins:
    def test_bestsource_and_vship_present(self, tmp_path: Path) -> None:
        bs = tmp_path / "tools" / "BestSource.dll"
        vs = tmp_path / "tools" / "libvship.dll"
        config = _write_config(
            tmp_path,
            tools_override={"bestsource": str(bs), "vship": str(vs)},
        )
        bs.touch()
        vs.touch()
        tp = load_config(config)
        assert tp.bestsource == bs
        assert tp.vship == vs

    def test_absent_defaults_none(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path)
        tp = load_config(config)
        assert tp.bestsource is None
        assert tp.vship is None

    def test_vship_path_missing_raises(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "tools" / "no_vship.dll")
        config = _write_config(tmp_path, tools_override={"vship": bogus})
        with pytest.raises(FileNotFoundError, match="vship"):
            load_config(config)


class TestAppdataFallback:
    def test_finds_config_in_appdata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(data_dir)

        appdata_dir = tmp_path / "appdata" / "furnace"
        appdata_dir.mkdir(parents=True)
        (appdata_dir / "furnace.toml").write_bytes(config.read_bytes())

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)
        fake_root = tmp_path / "fake_root" / "pkg"
        fake_root.mkdir(parents=True)
        monkeypatch.setattr(furnace.config, "__file__", str(fake_root / "config.py"))
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        tp = load_config()
        assert isinstance(tp, ToolPaths)
        assert tp.ffmpeg == data_dir / "tools" / "ffmpeg.exe"


class TestVersionSync:
    def test_package_version_matches_pyproject(self) -> None:
        import tomllib

        import furnace

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as f:
            declared = tomllib.load(f)["project"]["version"]
        assert declared == furnace.VERSION
