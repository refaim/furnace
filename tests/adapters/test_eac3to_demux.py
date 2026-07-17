from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from furnace.adapters.eac3to import Eac3toAdapter


class TestDemuxTitlePathHandling:
    def test_demux_title_passes_absolute_disc_path(self, tmp_path: Path) -> None:
        disc_root = tmp_path / "src2" / "MOVIE_BDCLUB"
        (disc_root / "BDMV").mkdir(parents=True)
        output_dir = tmp_path / "out" / "title_1"
        output_dir.mkdir(parents=True)

        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            relative_disc_path = Path("src2") / "MOVIE_BDCLUB" / "BDMV"
            assert not relative_disc_path.is_absolute()

            captured: dict[str, object] = {}

            def fake_run_tool(
                cmd: list[str | Path],
                on_output: object = None,
                on_progress_line: object = None,
                log_path: object = None,
                cwd: object = None,
            ) -> tuple[int, str]:
                captured["cmd"] = [str(c) for c in cmd]
                captured["cwd"] = cwd
                return (0, "")

            adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"))

            with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
                adapter.demux_title(
                    relative_disc_path,
                    title_num=1,
                    output_dir=output_dir,
                )

            cmd = captured["cmd"]
            assert isinstance(cmd, list)
            disc_path_arg = Path(cmd[1])
            assert disc_path_arg.is_absolute(), (
                f"disc_path passed to eac3to must be absolute so that cwd=output_dir "
                f"does not break resolution; got {disc_path_arg}"
            )
            assert disc_path_arg.resolve() == relative_disc_path.resolve()
        finally:
            os.chdir(original_cwd)
