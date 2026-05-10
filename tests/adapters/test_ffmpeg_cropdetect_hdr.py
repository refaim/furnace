"""Tests for the HDR-aware filter chain built by ``FFmpegAdapter.detect_crop``.

Mocks ``subprocess.run`` so no ffmpeg binary is invoked -- we just inspect the
``-vf`` argument the adapter constructs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

_SDR_VF = "cropdetect=24:16:0"
_SDR_VF_INTERLACED = "yadif,cropdetect=24:16:0"

_PQ_CHAIN = (
    "zscale=tin=smpte2084:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=24:16:0"
)
_HLG_CHAIN = (
    "zscale=tin=arib-std-b67:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=24:16:0"
)
_PQ_CHAIN_INTERLACED = "yadif," + _PQ_CHAIN


def _captured_vf(call_args_list: list[Any], point_idx: int = 0) -> str:
    """Pluck the value passed after ``-vf`` in the call's positional cmd list."""
    cmd = call_args_list[point_idx].args[0]
    vf_idx = cmd.index("-vf")
    return str(cmd[vf_idx + 1])


@pytest.mark.parametrize(
    ("interlaced", "hdr_transfer", "expected_vf"),
    [
        (False, None, _SDR_VF),
        (True, None, _SDR_VF_INTERLACED),
        (False, "smpte2084", _PQ_CHAIN),
        (False, "arib-std-b67", _HLG_CHAIN),
        (True, "smpte2084", _PQ_CHAIN_INTERLACED),
    ],
)
def test_detect_crop_filter_chain(
    interlaced: bool, hdr_transfer: str | None, expected_vf: str,
) -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        adapter.detect_crop(
            Path("x.mkv"),
            duration_s=1000.0,
            interlaced=interlaced,
            is_dvd=False,
            hdr_transfer=hdr_transfer,
        )
    # All 10 sample-point invocations should use the same -vf.
    assert mock_run.call_count == 10
    for i in range(10):
        assert _captured_vf(mock_run.call_args_list, i) == expected_vf
