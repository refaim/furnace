from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolPaths:
    ffmpeg: Path
    ffprobe: Path
    mkvmerge: Path
    mkvpropedit: Path
    mkclean: Path
    eac3to: Path
    qaac64: Path
    mpv: Path


def load_config(config_path: Path | None = None) -> ToolPaths:
    """Load TOML config. Search order:
    1. Explicit path (if provided)
    2. furnace.toml in CWD
    3. %APPDATA%\\furnace\\furnace.toml
    Validates that all tool paths exist.
    """
    searched: list[Path] = []

    def try_load(path: Path) -> dict[str, Any] | None:
        searched.append(path)
        if path.is_file():
            with path.open("rb") as f:
                return tomllib.load(f)
        return None

    data: dict[str, Any] | None = None

    if config_path is not None:
        data = try_load(config_path)
        if data is None:
            raise FileNotFoundError(
                f"Config file not found at explicit path: {config_path}"
            )
    else:
        # Try CWD
        data = try_load(Path.cwd() / "furnace.toml")

        # Try %APPDATA%\furnace\furnace.toml
        if data is None:
            appdata = os.environ.get("APPDATA")
            if appdata:
                data = try_load(Path(appdata) / "furnace" / "furnace.toml")

    if data is None:
        searched_str = "\n  ".join(str(p) for p in searched)
        raise FileNotFoundError(
            f"No furnace.toml config found. Searched:\n  {searched_str}"
        )

    tools_section: dict[str, Any] = data.get("tools", {})

    tool_names = ("ffmpeg", "ffprobe", "mkvmerge", "mkvpropedit", "mkclean", "eac3to", "qaac64", "mpv")
    resolved: dict[str, Path] = {}

    for name in tool_names:
        if name not in tools_section:
            raise KeyError(f"Missing required key [tools].{name} in config")
        tool_path = Path(tools_section[name])
        if not tool_path.exists():
            raise FileNotFoundError(
                f"Tool '{name}' not found at path: {tool_path}"
            )
        resolved[name] = tool_path

    return ToolPaths(
        ffmpeg=resolved["ffmpeg"],
        ffprobe=resolved["ffprobe"],
        mkvmerge=resolved["mkvmerge"],
        mkvpropedit=resolved["mkvpropedit"],
        mkclean=resolved["mkclean"],
        eac3to=resolved["eac3to"],
        qaac64=resolved["qaac64"],
        mpv=resolved["mpv"],
    )
