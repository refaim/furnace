from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

_SDR_VF = "format=yuv420p,cropdetect=40:2:0"
_SDR_VF_INTERLACED = "yadif,format=yuv420p,cropdetect=40:2:0"

_PQ_CHAIN = (
    "zscale=tin=smpte2084:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=40:2:0"
)
_HLG_CHAIN = (
    "zscale=tin=arib-std-b67:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=40:2:0"
)
_PQ_CHAIN_INTERLACED = "yadif," + _PQ_CHAIN


def _cropdetect_vfs(call_args_list: list[Any]) -> list[str]:
    vfs: list[str] = []
    for call in call_args_list:
        cmd = list(call.args[0])
        vf = str(cmd[cmd.index("-vf") + 1])
        if "cropdetect" in vf:
            vfs.append(vf)
    return vfs


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
    interlaced: bool,
    hdr_transfer: str | None,
    expected_vf: str,
) -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.stdout = b""
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=fake_result,
    ) as mock_run:
        adapter.detect_crop(
            Path("x.mkv"),
            duration_s=1000.0,
            interlaced=interlaced,
            is_dvd=False,
            hdr_transfer=hdr_transfer,
        )
    vfs = _cropdetect_vfs(mock_run.call_args_list)
    assert len(vfs) == 20
    assert vfs == [expected_vf] * 20


@pytest.mark.parametrize("hdr_transfer", [None, "smpte2084", "arib-std-b67"])
def test_cropdetect_downconverts_to_8bit_before_detect(hdr_transfer: str | None) -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.stdout = b""
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=fake_result,
    ) as mock_run:
        adapter.detect_crop(Path("x.mkv"), duration_s=1000.0, hdr_transfer=hdr_transfer)
    filters = _cropdetect_vfs(mock_run.call_args_list)[0].split(",")
    assert filters.count("format=yuv420p") == 1
    assert filters.index("format=yuv420p") == filters.index("cropdetect=40:2:0") - 1
