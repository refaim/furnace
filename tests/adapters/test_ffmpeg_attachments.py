from pathlib import Path
from unittest.mock import patch

from furnace.adapters.ffmpeg import FFmpegAdapter


def test_extract_attachment_builds_dump_command(tmp_path: Path) -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"), log_dir=tmp_path)
    source = Path("source.mkv")
    output = tmp_path / "font.ttf"
    captured: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> tuple[int, str]:
        captured.extend(cmd)
        return 0, ""

    with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run):
        result = adapter.extract_attachment(source, 4, output)

    assert result == 0
    assert "-dump_attachment:4" in captured
    assert str(output) in captured
    assert "-t" in captured
    assert captured[captured.index("-t") + 1] == "0"
    assert adapter._log_dir == tmp_path
