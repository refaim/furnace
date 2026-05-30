"""Tests for ``FFmpegAdapter.first_second_rms_db``.

The adapter runs ``ffmpeg -t 1 -af astats=metadata=1 -f null -`` on the
input file and parses the ``RMS level dB`` line from stderr. We patch
``subprocess.run`` so no real ffmpeg is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))


def test_first_second_rms_db_parses_silence() -> None:
    """ffmpeg stderr reporting RMS level near -inf → returned float < -50."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = (
        "[Parsed_astats_0 @ 0x55] Channel: 1\n"
        "[Parsed_astats_0 @ 0x55] RMS level dB: -102.479232\n"
        "[Parsed_astats_0 @ 0x55] Peak level dB: -90.308734\n"
    )
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake) as run:
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is not None
    assert abs(rms - (-102.479232)) < 1e-6
    run.assert_called_once()


def test_first_second_rms_db_parses_loud_audio() -> None:
    """ffmpeg stderr with a normal loud RMS level → returned as float."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = (
        "[Parsed_astats_0 @ 0x77] RMS level dB: -25.644711\n"
        "[Parsed_astats_0 @ 0x77] Peak level dB: -17.481235\n"
    )
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is not None
    assert abs(rms - (-25.644711)) < 1e-6


def test_first_second_rms_db_command_arguments(tmp_path: Path) -> None:
    """Verify the command line — input path, ``-t 1``, astats filter, null sink."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = "RMS level dB: -50.000000\n"
    audio_path = tmp_path / "audio.dts"
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake) as run:
        _adapter().first_second_rms_db(audio_path)

    cmd = run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    i_idx = cmd.index("-i")
    assert cmd[i_idx + 1] == str(audio_path)
    assert "-t" in cmd
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "1"
    assert "-af" in cmd
    af_idx = cmd.index("-af")
    assert "astats" in cmd[af_idx + 1]
    assert "-f" in cmd
    f_idx = cmd.index("-f")
    assert cmd[f_idx + 1] == "null"
    # Output sink is "-"
    assert cmd[-1] == "-"


def test_first_second_rms_db_returns_none_on_nonzero_returncode() -> None:
    fake = MagicMock()
    fake.returncode = 1
    fake.stderr = "ffmpeg: some error\n"
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is None


def test_first_second_rms_db_returns_none_when_no_rms_line() -> None:
    """rc=0 but stderr lacks the RMS line → returns None."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = "[info] some other ffmpeg blather\n"
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is None


def test_first_second_rms_db_returns_none_on_unparseable_rms_value() -> None:
    """RMS line present but value is not a float → returns None."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = "[Parsed_astats_0 @ 0x77] RMS level dB: bogus\n"
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is None


def test_first_second_rms_db_handles_minus_inf() -> None:
    """ffmpeg sometimes prints ``-inf`` for fully silent input."""
    fake = MagicMock()
    fake.returncode = 0
    fake.stderr = "[Parsed_astats_0 @ 0x77] RMS level dB: -inf\n"
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is not None
    assert rms == float("-inf")


def test_first_second_rms_db_returns_none_when_subprocess_raises() -> None:
    """If subprocess.run raises (e.g. OSError on missing ffmpeg), return None
    rather than propagating — the caller (disc_demuxer) treats None as
    ``cannot classify`` and skips --sync. That's the correct safe fallback.
    """
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        side_effect=OSError("no such binary"),
    ):
        rms = _adapter().first_second_rms_db(Path("audio.dts"))
    assert rms is None
