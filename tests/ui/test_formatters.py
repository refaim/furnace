"""Pure-function tests for ALL formatting helpers across the UI layer.

Covers:
- furnace.ui.fmt.fmt_size
- furnace.ui.run_tui: _fmt_time, _fmt_bitrate, _channel_layout_short,
    _audio_step_label, _sub_step_label, _build_steps, _build_source_text,
    _build_target_text, _sub_target_label
- furnace.ui.tui: _fmt_duration, _fmt_subtitle_track
"""
from __future__ import annotations

from furnace.core.models import (
    AudioAction,
    CropRect,
    DownmixMode,
    SubtitleAction,
    TrackType,
)
from furnace.ui.fmt import fmt_size
from furnace.ui.run_tui import (
    _audio_step_label,
    _build_source_text,
    _build_steps,
    _build_target_text,
    _channel_layout_short,
    _fmt_bitrate,
    _fmt_time,
    _sub_step_label,
    _sub_target_label,
)
from furnace.ui.tui import _fmt_duration, _fmt_subtitle_track
from tests.conftest import (
    make_audio_instruction,
    make_job,
    make_subtitle_instruction,
    make_track,
    make_video_params,
)


class TestFmtSize:
    def test_none_returns_question_mark(self) -> None:
        assert fmt_size(None) == "?"

    def test_zero_returns_question_mark(self) -> None:
        assert fmt_size(0) == "?"

    def test_one_megabyte(self) -> None:
        assert fmt_size(1024 * 1024) == "1 MB"

    def test_large_value_with_comma_grouping(self) -> None:
        result = fmt_size(5 * 1024 * 1024 * 1024)
        assert result == "5,120 MB"

    def test_fractional_megabyte_rounded(self) -> None:
        result = fmt_size(1_572_864)
        assert result == "2 MB"

    def test_small_value_below_one_mb(self) -> None:
        result = fmt_size(512_000)
        assert result == "0 MB"

    def test_exact_100_mb(self) -> None:
        result = fmt_size(100 * 1024 * 1024)
        assert result == "100 MB"


class TestFmtTime:
    def test_zero_seconds(self) -> None:
        assert _fmt_time(0.0) == "0:00"

    def test_seconds_only(self) -> None:
        assert _fmt_time(45.0) == "0:45"

    def test_minutes_and_seconds(self) -> None:
        assert _fmt_time(125.0) == "2:05"

    def test_exact_one_hour(self) -> None:
        assert _fmt_time(3600.0) == "1:00:00"

    def test_hours_minutes_seconds(self) -> None:
        assert _fmt_time(3661.0) == "1:01:01"

    def test_fractional_seconds_truncated(self) -> None:
        assert _fmt_time(65.9) == "1:05"

    def test_large_hours(self) -> None:
        assert _fmt_time(36000.0) == "10:00:00"

    def test_59_minutes_59_seconds(self) -> None:
        assert _fmt_time(3599.0) == "59:59"


class TestFmtBitrate:
    def test_none_returns_empty(self) -> None:
        assert _fmt_bitrate(None) == ""

    def test_zero_returns_empty(self) -> None:
        assert _fmt_bitrate(0) == ""

    def test_kbps_range(self) -> None:
        assert _fmt_bitrate(192_000) == "192kbps"

    def test_high_bitrate(self) -> None:
        assert _fmt_bitrate(6_000_000) == "6000kbps"

    def test_sub_1000_bps(self) -> None:
        assert _fmt_bitrate(500) == "500bps"

    def test_exact_1000_bps(self) -> None:
        assert _fmt_bitrate(1000) == "1kbps"


class TestChannelLayoutShort:
    def test_none_returns_empty(self) -> None:
        assert _channel_layout_short(None) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert _channel_layout_short("") == ""

    def test_plain_layout(self) -> None:
        assert _channel_layout_short("5.1") == "5.1"

    def test_layout_with_parenthesized_suffix(self) -> None:
        assert _channel_layout_short("5.1(side)") == "5.1"

    def test_stereo(self) -> None:
        assert _channel_layout_short("stereo") == "stereo"

    def test_7_1_with_suffix(self) -> None:
        assert _channel_layout_short("7.1(wide)") == "7.1"


class TestAudioStepLabel:
    def test_copy_single_track(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=2)
        result = _audio_step_label(instr, 0, 1)
        assert result == "Copy audio (AAC 2.0)"

    def test_copy_multiple_tracks_shows_number(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=6)
        result = _audio_step_label(instr, 0, 2)
        assert result == "Copy audio 1 (AAC 5.1)"

    def test_denorm(self) -> None:
        instr = make_audio_instruction(action=AudioAction.DENORM, codec_name="ac3", channels=6)
        result = _audio_step_label(instr, 0, 1)
        assert result == "Denorm audio (AC3 5.1)"

    def test_decode_encode(self) -> None:
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, codec_name="truehd", channels=8,
        )
        result = _audio_step_label(instr, 0, 1)
        assert result == "Recode audio (TRUEHD -> AAC)"

    def test_ffmpeg_encode(self) -> None:
        instr = make_audio_instruction(
            action=AudioAction.FFMPEG_ENCODE, codec_name="opus", channels=6,
        )
        result = _audio_step_label(instr, 0, 1)
        assert result == "Recode audio (OPUS -> AAC)"

    def test_7_1_channel_map(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="dts", channels=8)
        result = _audio_step_label(instr, 0, 1)
        assert "7.1" in result

    def test_mono_channel_map(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=1)
        result = _audio_step_label(instr, 0, 1)
        assert "1.0" in result

    def test_unknown_channel_count(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=4)
        result = _audio_step_label(instr, 0, 1)
        assert "4ch" in result

    def test_none_channels_omits_layout(self) -> None:
        instr = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=None)
        result = _audio_step_label(instr, 0, 1)
        assert result == "Copy audio (AAC)"

    def test_second_of_three_tracks(self) -> None:
        instr = make_audio_instruction(action=AudioAction.DENORM, codec_name="dts", channels=6)
        result = _audio_step_label(instr, 1, 3)
        assert result == "Denorm audio 2 (DTS 5.1)"


class TestSubStepLabel:
    def test_copy_single_track(self) -> None:
        instr = make_subtitle_instruction(
            action=SubtitleAction.COPY, codec_name="hdmv_pgs_subtitle",
        )
        result = _sub_step_label(instr, 0, 1)
        assert result == "Copy subs (HDMV_PGS_SUBTITLE)"

    def test_copy_recode(self) -> None:
        instr = make_subtitle_instruction(action=SubtitleAction.COPY_RECODE, codec_name="subrip")
        result = _sub_step_label(instr, 0, 1)
        assert result == "Recode subs (SUBRIP -> UTF-8)"

    def test_multiple_tracks_shows_number(self) -> None:
        instr = make_subtitle_instruction(
            action=SubtitleAction.COPY, codec_name="hdmv_pgs_subtitle",
        )
        result = _sub_step_label(instr, 1, 3)
        assert result == "Copy subs 2 (HDMV_PGS_SUBTITLE)"

    def test_recode_multiple_tracks(self) -> None:
        instr = make_subtitle_instruction(action=SubtitleAction.COPY_RECODE, codec_name="ass")
        result = _sub_step_label(instr, 0, 2)
        assert result == "Recode subs 1 (ASS -> UTF-8)"


class TestSubTargetLabel:
    def test_copy_shows_codec_and_language(self) -> None:
        instr = make_subtitle_instruction(
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            language="eng",
        )
        assert _sub_target_label(instr) == "HDMV_PGS_SUBTITLE eng"

    def test_copy_recode_shows_utf8_tag(self) -> None:
        instr = make_subtitle_instruction(
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            language="rus",
        )
        assert _sub_target_label(instr) == "SUBRIP rus (UTF-8)"


class TestBuildSteps:
    def test_basic_job_steps(self) -> None:
        job = make_job()
        steps = _build_steps(job)
        assert len(steps) == 6
        assert steps[-4] == "Encode video"
        assert steps[-3] == "Assemble MKV"
        assert steps[-2] == "Set metadata"
        assert steps[-1] == "Optimize index"

    def test_vmaf_adds_step(self) -> None:
        job = make_job()
        steps = _build_steps(job, vmaf_enabled=True)
        assert steps[-1] == "VMAF"
        assert len(steps) == 7

    def test_no_audio_no_subs(self) -> None:
        job = make_job(audio=[], subtitles=[])
        steps = _build_steps(job)
        assert len(steps) == 4

    def test_multiple_audio_tracks(self) -> None:
        audio = [
            make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=2),
            make_audio_instruction(action=AudioAction.DENORM, codec_name="ac3", channels=6),
        ]
        job = make_job(audio=audio)
        steps = _build_steps(job)
        assert "Copy audio 1" in steps[0]
        assert "Denorm audio 2" in steps[1]

    def test_multiple_subtitle_tracks(self) -> None:
        subs = [
            make_subtitle_instruction(
                action=SubtitleAction.COPY, codec_name="hdmv_pgs_subtitle",
            ),
            make_subtitle_instruction(action=SubtitleAction.COPY_RECODE, codec_name="subrip"),
        ]
        job = make_job(subtitles=subs)
        steps = _build_steps(job)
        assert len(steps) == 7
        assert "Copy subs 1" in steps[1]
        assert "Recode subs 2" in steps[2]


class TestBuildSourceText:
    def test_basic_source(self) -> None:
        vp = make_video_params(
            source_codec="h264",
            source_width=1920,
            source_height=1080,
            source_bitrate=8_500_000,
        )
        job = make_job(video_params=vp)
        text = _build_source_text(job)
        assert "H264" in text
        assert "1920x1080" in text
        assert "8500kbps" in text

    def test_no_bitrate_omits_video_bitrate(self) -> None:
        vp = make_video_params(source_codec="hevc", source_bitrate=0)
        job = make_job(video_params=vp, audio=[], subtitles=[])
        text = _build_source_text(job)
        video_line = text.split("\n")[0]
        assert "kbps" not in video_line
        assert "HEVC" in video_line

    def test_audio_line_codec_and_channels(self) -> None:
        audio = [make_audio_instruction(
            codec_name="dts", channels=6, bitrate=1_500_000,
        )]
        job = make_job(audio=audio)
        text = _build_source_text(job)
        assert "Audio:" in text
        assert "DTS" in text
        assert "5.1" in text
        assert "1500kbps" in text

    def test_multiple_audio_indented(self) -> None:
        audio = [
            make_audio_instruction(codec_name="dts", channels=6, bitrate=1_500_000),
            make_audio_instruction(codec_name="aac", channels=2, bitrate=192_000),
        ]
        job = make_job(audio=audio)
        text = _build_source_text(job)
        lines = text.split("\n")
        audio_lines = [line for line in lines if "DTS" in line or "AAC" in line]
        assert len(audio_lines) == 2
        assert audio_lines[0].startswith("Audio:")
        assert audio_lines[1].startswith("      ")

    def test_subtitle_line(self) -> None:
        sub = [make_subtitle_instruction(
            codec_name="hdmv_pgs_subtitle", language="eng",
        )]
        job = make_job(subtitles=sub)
        text = _build_source_text(job)
        assert "Subs:" in text
        assert "HDMV_PGS_SUBTITLE" in text
        assert "eng" in text

    def test_source_size_shown_when_positive(self) -> None:
        job = make_job(source_size=1_048_576)
        text = _build_source_text(job)
        assert "Size:" in text
        assert "1 MB" in text

    def test_source_size_hidden_when_zero(self) -> None:
        job = make_job(source_size=0)
        text = _build_source_text(job)
        assert "Size:" not in text

    def test_audio_with_no_channels(self) -> None:
        audio = [make_audio_instruction(codec_name="aac", channels=None, bitrate=128_000)]
        job = make_job(audio=audio)
        text = _build_source_text(job)
        assert "AAC" in text
        assert "128kbps" in text

    def test_audio_with_no_bitrate(self) -> None:
        audio = [make_audio_instruction(codec_name="dts", channels=6, bitrate=None)]
        job = make_job(audio=audio)
        text = _build_source_text(job)
        assert "DTS" in text
        assert "5.1" in text


class TestBuildTargetText:
    def test_basic_target(self) -> None:
        vp = make_video_params(source_width=1920, source_height=1080, cq=25)
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "HEVC" in text
        assert "1920x1080" in text
        assert "CQ25" in text

    def test_crop_changes_resolution(self) -> None:
        crop = CropRect(w=1920, h=800, x=0, y=140)
        vp = make_video_params(
            source_width=1920, source_height=1080, cq=25, crop=crop,
        )
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "1920x800" in text

    def test_sar_corrects_resolution(self) -> None:
        vp = make_video_params(
            source_width=720, source_height=480, cq=22,
            sar_num=64, sar_den=45,
        )
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "1024x480" in text

    def test_audio_target_label_in_text(self) -> None:
        audio = [make_audio_instruction(
            action=AudioAction.DENORM, codec_name="ac3", channels=6,
        )]
        job = make_job(audio=audio)
        text = _build_target_text(job)
        assert "Audio:" in text
        assert "AC3" in text
        assert "denorm" in text

    def test_subtitle_target_label_in_text(self) -> None:
        sub = [make_subtitle_instruction(
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            language="eng",
        )]
        job = make_job(subtitles=sub)
        text = _build_target_text(job)
        assert "Subs:" in text
        assert "SUBRIP eng (UTF-8)" in text

    def test_multiple_audio_targets(self) -> None:
        audio = [
            make_audio_instruction(action=AudioAction.COPY, codec_name="aac", channels=2),
            make_audio_instruction(action=AudioAction.DENORM, codec_name="dts", channels=6),
        ]
        job = make_job(audio=audio)
        text = _build_target_text(job)
        lines = text.split("\n")
        audio_lines = [line for line in lines if "AAC" in line or "DTS" in line]
        assert len(audio_lines) == 2
        assert audio_lines[0].startswith("Audio:")
        assert audio_lines[1].startswith("      ")

    def test_downmix_stereo_shows_2_0(self) -> None:
        audio = [make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="truehd",
            channels=8,
            downmix=DownmixMode.STEREO,
        )]
        job = make_job(audio=audio)
        text = _build_target_text(job)
        assert "2.0" in text
        assert "7.1" not in text

    def test_no_crop_uses_source_dimensions(self) -> None:
        vp = make_video_params(source_width=3840, source_height=2160, cq=31)
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "3840x2160" in text

    def test_crop_with_sar(self) -> None:
        crop = CropRect(w=704, h=464, x=8, y=8)
        vp = make_video_params(
            source_width=720, source_height=480, cq=22,
            sar_num=64, sar_den=45, crop=crop,
        )
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "1001x464" in text


class TestFmtDuration:
    def test_zero(self) -> None:
        assert _fmt_duration(0.0) == "0:00"

    def test_seconds_only(self) -> None:
        assert _fmt_duration(30.0) == "0:30"

    def test_minutes_and_seconds(self) -> None:
        assert _fmt_duration(90.0) == "1:30"

    def test_exact_hour(self) -> None:
        assert _fmt_duration(3600.0) == "1:00:00"

    def test_hours_minutes_seconds(self) -> None:
        assert _fmt_duration(5400.0) == "1:30:00"

    def test_fractional_truncated(self) -> None:
        assert _fmt_duration(61.7) == "1:01"

    def test_two_hours(self) -> None:
        assert _fmt_duration(7261.0) == "2:01:01"


class TestFmtSubtitleTrack:
    def test_selected_basic(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="hdmv_pgs_subtitle",
            language="eng",
            is_forced=False,
            title="",
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert result.startswith("\\[x]")
        assert "HDMV_PGS_SUBTITLE" in result
        assert "eng" in result

    def test_unselected(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            language="rus",
        )
        result = _fmt_subtitle_track(track, selected=False)
        assert result.startswith("\\[ ]")

    def test_forced_flag(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="hdmv_pgs_subtitle",
            language="eng",
            is_forced=True,
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert "[FORCED]" in result

    def test_not_forced_no_flag(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="hdmv_pgs_subtitle",
            language="eng",
            is_forced=False,
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert "FORCED" not in result

    def test_title_shown(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            language="eng",
            title="SDH",
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert "'SDH'" in result

    def test_no_title_omitted(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            language="eng",
            title="",
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert "'" not in result

    def test_und_language_padded(self) -> None:
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            language="",
        )
        result = _fmt_subtitle_track(track, selected=True)
        assert "und " in result
