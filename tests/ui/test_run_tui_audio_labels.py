"""Pure-function tests for `_audio_target_label` in `furnace.ui.run_tui`.

These tests specifically pin down the target-frame audio label format, which
must show the OUTPUT channel layout (after any downmix) so the user can see
at a glance what the processed file will contain.
"""
from __future__ import annotations

from furnace.core.models import AudioAction, AudioInstruction, DownmixMode
from furnace.ui.run_tui import _audio_target_label
from tests.conftest import make_audio_instruction


def _instr(
    *,
    action: AudioAction,
    codec_name: str = "dts",
    channels: int | None = 6,
    downmix: DownmixMode | None = None,
) -> AudioInstruction:
    return make_audio_instruction(
        action=action,
        codec_name=codec_name,
        channels=channels,
        bitrate=448_000,
        downmix=downmix,
    )


class TestAudioTargetLabelCopy:
    def test_copy_5_1(self) -> None:
        label = _audio_target_label(_instr(action=AudioAction.COPY, codec_name="aac", channels=6))
        assert "5.1" in label
        assert "AAC" in label
        assert "copy" in label

    def test_copy_stereo(self) -> None:
        label = _audio_target_label(_instr(action=AudioAction.COPY, codec_name="aac", channels=2))
        assert "2.0" in label
        assert "AAC" in label


class TestAudioTargetLabelDenorm:
    def test_denorm_5_1(self) -> None:
        label = _audio_target_label(_instr(action=AudioAction.DENORM, codec_name="ac3", channels=6))
        assert "5.1" in label
        assert "AC3" in label
        assert "denorm" in label


class TestAudioTargetLabelDecodeEncodeNoDownmix:
    """Without downmix, DECODE_ENCODE preserves the source channel count."""

    def test_truehd_7_1_no_downmix(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.DECODE_ENCODE, codec_name="truehd", channels=8)
        )
        assert "AAC" in label
        assert "7.1" in label
        assert "TRUEHD" in label

    def test_dts_ma_5_1_no_downmix(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.DECODE_ENCODE, codec_name="dts", channels=6)
        )
        assert "AAC" in label
        assert "5.1" in label

    def test_flac_2_0_no_downmix(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.DECODE_ENCODE, codec_name="flac", channels=2)
        )
        assert "AAC" in label
        assert "2.0" in label


class TestAudioTargetLabelDecodeEncodeDownmix:
    """With downmix, target channels differ from source channels."""

    def test_7_1_downmix_to_stereo(self) -> None:
        label = _audio_target_label(
            _instr(
                action=AudioAction.DECODE_ENCODE,
                codec_name="truehd",
                channels=8,
                downmix=DownmixMode.STEREO,
            )
        )
        assert "AAC" in label
        assert "2.0" in label
        assert "7.1" not in label  # must not show the source layout

    def test_5_1_downmix_to_stereo(self) -> None:
        label = _audio_target_label(
            _instr(
                action=AudioAction.DECODE_ENCODE,
                codec_name="dts",
                channels=6,
                downmix=DownmixMode.STEREO,
            )
        )
        assert "AAC" in label
        assert "2.0" in label
        assert "5.1" not in label

    def test_7_1_downmix_to_5_1(self) -> None:
        label = _audio_target_label(
            _instr(
                action=AudioAction.DECODE_ENCODE,
                codec_name="truehd",
                channels=8,
                downmix=DownmixMode.DOWN6,
            )
        )
        assert "AAC" in label
        assert "5.1" in label
        assert "7.1" not in label


class TestAudioTargetLabelFfmpegEncode:
    def test_opus_5_1_no_downmix(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.FFMPEG_ENCODE, codec_name="opus", channels=6)
        )
        assert "AAC" in label
        assert "5.1" in label

    def test_opus_5_1_downmix_stereo(self) -> None:
        label = _audio_target_label(
            _instr(
                action=AudioAction.FFMPEG_ENCODE,
                codec_name="opus",
                channels=6,
                downmix=DownmixMode.STEREO,
            )
        )
        assert "AAC" in label
        assert "2.0" in label
        assert "5.1" not in label


class TestAudioTargetLabelUnknownChannels:
    """If channels is None, no layout is rendered — but the rest still shows."""

    def test_copy_unknown_channels(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.COPY, codec_name="aac", channels=None)
        )
        assert "AAC" in label
        assert "copy" in label

    def test_decode_encode_unknown_channels_no_downmix(self) -> None:
        label = _audio_target_label(
            _instr(action=AudioAction.DECODE_ENCODE, codec_name="truehd", channels=None)
        )
        assert "AAC" in label
