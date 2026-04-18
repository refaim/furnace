"""Pure-function tests for `_fmt_audio_track`."""

from __future__ import annotations

from pathlib import Path

from furnace.core.audio_profile import AudioMetrics, AudioProfile, Verdict
from furnace.core.models import AudioCodecId, DownmixMode, Track, TrackType
from furnace.ui.tui import TrackSelection, TrackSelectorScreen, _fmt_audio_track
from tests.conftest import make_movie, make_track, make_video_info

# ---------------------------------------------------------------------------
# Audio-profile helpers used by detector-tag and auto-preselect tests
# ---------------------------------------------------------------------------


def _fake_profile(suggested: DownmixMode | None) -> AudioProfile:
    metrics = AudioMetrics(
        channels=6,
        rms_l=-50.0, rms_r=-47.0, rms_c=-28.0, rms_lfe=-75.0,
        rms_ls=-47.0, rms_rs=-49.0, rms_lb=None, rms_rb=None,
        corr_lr=0.4, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.2,
        corr_lb_ls=None, corr_rb_rs=None,
    )
    return AudioProfile(
        verdict=Verdict.FAKE, score=2, suggested=suggested,
        reasons=("LFE is dead", "center is way louder than everything else"),
        metrics=metrics,
    )


def _suspicious_profile() -> AudioProfile:
    metrics = AudioMetrics(
        channels=6,
        rms_l=-44.0, rms_r=-42.5, rms_c=-40.6, rms_lfe=-59.7,
        rms_ls=-50.8, rms_rs=-49.7, rms_lb=None, rms_rb=None,
        corr_lr=0.956, corr_ls_l=-0.008, corr_rs_r=-0.003, corr_ls_rs=0.863,
        corr_lb_ls=None, corr_rb_rs=None,
    )
    return AudioProfile(
        verdict=Verdict.SUSPICIOUS, score=1, suggested=DownmixMode.STEREO,
        reasons=("left and right surrounds carry the same signal",),
        metrics=metrics,
    )


def _real_profile() -> AudioProfile:
    metrics = AudioMetrics(
        channels=6,
        rms_l=-20.0, rms_r=-20.5, rms_c=-25.0, rms_lfe=-30.0,
        rms_ls=-22.0, rms_rs=-22.5, rms_lb=None, rms_rb=None,
        corr_lr=0.3, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.2,
        corr_lb_ls=None, corr_rb_rs=None,
    )
    return AudioProfile(
        verdict=Verdict.REAL, score=0, suggested=None,
        reasons=(), metrics=metrics,
    )


def _t(channels: int | None = 6, codec: str = "dts", layout: str = "5.1") -> Track:
    return make_track(
        index=1,
        track_type=TrackType.AUDIO,
        codec_name=codec,
        codec_id=AudioCodecId.DTS_MA if codec == "dts" else AudioCodecId.AAC_LC,
        language="eng",
        title="Main",
        is_default=True,
        channels=channels,
        channel_layout=layout,
        bitrate=3_500_000,
    )


class TestFmtAudioTrackDownmixTag:
    def test_no_downmix_has_no_tag(self) -> None:
        line = _fmt_audio_track(_t(), selected=True, downmix=None)
        assert "[->" not in line

    def test_stereo_tag(self) -> None:
        line = _fmt_audio_track(_t(), selected=True, downmix=DownmixMode.STEREO)
        assert "[-> 2.0]" in line

    def test_mono_tag(self) -> None:
        line = _fmt_audio_track(_t(), selected=True, downmix=DownmixMode.MONO)
        assert "[-> 1.0]" in line

    def test_down6_tag(self) -> None:
        line = _fmt_audio_track(_t(channels=8, layout="7.1"), selected=True, downmix=DownmixMode.DOWN6)
        assert "[-> 5.1]" in line

    def test_unselected_still_formats(self) -> None:
        line = _fmt_audio_track(_t(), selected=False, downmix=DownmixMode.STEREO)
        assert "[-> 2.0]" in line
        # Unselected marker: a space inside the leading [x/ ]
        assert line.startswith("\\[ ]")

    def test_codec_and_layout_still_present(self) -> None:
        """Existing content (codec, layout, bitrate, title) still renders."""
        line = _fmt_audio_track(_t(), selected=True, downmix=DownmixMode.STEREO)
        assert "DTS" in line
        assert "5.1" in line
        assert "3500 kbps" in line
        assert "Main" in line

    def test_fmt_audio_track_no_channel_layout(self) -> None:
        """Track with no channel_layout renders codec without layout suffix."""
        track = make_track(
            index=1,
            track_type=TrackType.AUDIO,
            codec_name="aac",
            codec_id=AudioCodecId.AAC_LC,
            language="eng",
            title="",
            channels=2,
            channel_layout=None,
            bitrate=128_000,
        )
        line = _fmt_audio_track(track, selected=True, downmix=None)
        # Codec still present, but no layout (no "5.1" or trailing layout token).
        assert "AAC" in line
        # Bitrate still shown, so "128 kbps" should appear.
        assert "128 kbps" in line
        # No parenthesised or dotted layout token.
        assert "5.1" not in line
        assert "7.1" not in line

    def test_fmt_audio_track_no_bitrate(self) -> None:
        """Track with no bitrate renders without the kbps suffix."""
        track = make_track(
            index=1,
            track_type=TrackType.AUDIO,
            codec_name="aac",
            codec_id=AudioCodecId.AAC_LC,
            language="eng",
            title="",
            channels=2,
            channel_layout="stereo",
            bitrate=None,
        )
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "AAC" in line
        assert "stereo" in line
        assert "kbps" not in line


class TestTrackSelection:
    def test_default_empty_downmix(self) -> None:
        sel = TrackSelection(tracks=[], downmix={})
        assert sel.tracks == []
        assert sel.downmix == {}

    def test_with_downmix(self) -> None:
        t = _t()
        sel = TrackSelection(
            tracks=[t],
            downmix={(Path("/src/movie.mkv"), 1): DownmixMode.STEREO},
        )
        assert sel.tracks == [t]
        assert sel.downmix[(Path("/src/movie.mkv"), 1)] == DownmixMode.STEREO


class TestTrackSelectorDownmixLogic:
    """Pure-logic tests that exercise action_set_downmix without a running Textual app.

    This instantiates the class but calls the action method directly. Because the
    method touches self.query_one() via _refresh_item(), we monkeypatch _refresh_item
    to be a no-op for the duration of the test.
    """

    def make_screen(self, tracks: list[Track]) -> TrackSelectorScreen:
        movie = make_movie(
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                duration_s=120.0, bitrate=10_000_000,
            ),
            audio_tracks=tracks,
        )
        screen = TrackSelectorScreen(movie, tracks, TrackType.AUDIO)
        # Monkeypatch the method to a no-op for pure-logic tests; mypy can't model
        # method reassignment on an instance, so silence [method-assign] here.
        screen._refresh_item = lambda index: None  # type: ignore[method-assign]
        return screen

    def test_set_stereo_on_multichannel(self) -> None:
        screen = self.make_screen([_t(channels=8)])
        screen._cursor = 0
        screen.action_set_downmix("stereo")
        assert screen._downmix[0] == DownmixMode.STEREO

    def test_repress_clears_mode(self) -> None:
        screen = self.make_screen([_t(channels=8)])
        screen._cursor = 0
        screen.action_set_downmix("stereo")
        screen.action_set_downmix("stereo")
        assert screen._downmix[0] is None

    def test_down6_noop_on_5_1(self) -> None:
        screen = self.make_screen([_t(channels=6)])
        screen._cursor = 0
        screen.action_set_downmix("down6")
        assert screen._downmix[0] is None

    def test_down6_works_on_7_1(self) -> None:
        screen = self.make_screen([_t(channels=8)])
        screen._cursor = 0
        screen.action_set_downmix("down6")
        assert screen._downmix[0] == DownmixMode.DOWN6

    def test_stereo_noop_on_stereo_track(self) -> None:
        screen = self.make_screen([_t(channels=2)])
        screen._cursor = 0
        screen.action_set_downmix("stereo")
        assert screen._downmix[0] is None

    def test_stereo_noop_on_unknown_channels(self) -> None:
        screen = self.make_screen([_t(channels=None)])
        screen._cursor = 0
        screen.action_set_downmix("stereo")
        assert screen._downmix[0] is None

    def test_mono_on_5_1_sets_mono(self) -> None:
        screen = self.make_screen([_t(channels=6)])
        screen._cursor = 0
        screen.action_set_downmix("mono")
        assert screen._downmix[0] == DownmixMode.MONO

    def test_mono_repress_clears_mode(self) -> None:
        screen = self.make_screen([_t(channels=6)])
        screen._cursor = 0
        screen.action_set_downmix("mono")
        screen.action_set_downmix("mono")
        assert screen._downmix[0] is None

    def test_mono_on_stereo_sets_mono(self) -> None:
        screen = self.make_screen([_t(channels=2)])
        screen._cursor = 0
        screen.action_set_downmix("mono")
        assert screen._downmix[0] == DownmixMode.MONO

    def test_mono_noop_on_mono_track(self) -> None:
        screen = self.make_screen([_t(channels=1)])
        screen._cursor = 0
        screen.action_set_downmix("mono")
        assert screen._downmix[0] is None

    def test_mono_noop_on_unknown_channels(self) -> None:
        screen = self.make_screen([_t(channels=None)])
        screen._cursor = 0
        screen.action_set_downmix("mono")
        assert screen._downmix[0] is None


class TestFmtAudioTrackDetectorTag:
    """Detector-tag rendering in `_fmt_audio_track` (FAKE/SUSPICIOUS/REAL)."""

    def test_fmt_audio_track_shows_fake_tag_when_no_manual_downmix(self) -> None:
        track = _t()
        track.audio_profile = _fake_profile(DownmixMode.STEREO)
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "[FAKE -> 2.0]" in line

    def test_fmt_audio_track_shows_fake_mono_tag_for_mono_suggestion(self) -> None:
        track = _t()
        track.audio_profile = _fake_profile(DownmixMode.MONO)
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "[FAKE -> 1.0]" in line

    def test_fmt_audio_track_shows_fake_down6_tag(self) -> None:
        track = _t(channels=8, layout="7.1")
        track.audio_profile = _fake_profile(DownmixMode.DOWN6)
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "[FAKE -> 5.1]" in line

    def test_fmt_audio_track_shows_suspicious_tag(self) -> None:
        track = _t()
        track.audio_profile = _suspicious_profile()
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "[SUSPICIOUS]" in line

    def test_fmt_audio_track_manual_downmix_hides_detector_tag(self) -> None:
        track = _t()
        track.audio_profile = _fake_profile(DownmixMode.STEREO)
        line = _fmt_audio_track(track, selected=True, downmix=DownmixMode.STEREO)
        assert "[FAKE" not in line
        assert "[-> 2.0]" in line

    def test_fmt_audio_track_real_profile_shows_no_tag(self) -> None:
        track = _t()
        track.audio_profile = _real_profile()
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "FAKE" not in line
        assert "SUSPICIOUS" not in line

    def test_fmt_audio_track_no_profile_shows_no_tag(self) -> None:
        track = _t()
        # audio_profile defaults to None
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "FAKE" not in line
        assert "SUSPICIOUS" not in line

    def test_fmt_audio_track_fake_without_suggested_shows_no_tag(self) -> None:
        """FAKE verdict but suggested=None (defensive) must not render a FAKE tag."""
        track = _t()
        track.audio_profile = _fake_profile(None)
        line = _fmt_audio_track(track, selected=True, downmix=None)
        assert "FAKE" not in line
        assert "SUSPICIOUS" not in line


class TestTrackSelectorAutoPreselect:
    """Auto-preselection of detector-suggested downmix on FAKE tracks."""

    def make_screen(self, tracks: list[Track]) -> TrackSelectorScreen:
        movie = make_movie(
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                duration_s=120.0, bitrate=10_000_000,
            ),
            audio_tracks=tracks,
        )
        return TrackSelectorScreen(movie, tracks, TrackType.AUDIO)

    def test_fake_stereo_on_5_1_auto_preselects_stereo(self) -> None:
        track = _t(channels=6)
        track.audio_profile = _fake_profile(DownmixMode.STEREO)
        screen = self.make_screen([track])
        assert screen._downmix[0] == DownmixMode.STEREO

    def test_fake_mono_on_5_1_auto_preselects_mono(self) -> None:
        track = _t(channels=6)
        track.audio_profile = _fake_profile(DownmixMode.MONO)
        screen = self.make_screen([track])
        assert screen._downmix[0] == DownmixMode.MONO

    def test_fake_down6_on_7_1_auto_preselects_down6(self) -> None:
        track = _t(channels=8, layout="7.1")
        track.audio_profile = _fake_profile(DownmixMode.DOWN6)
        screen = self.make_screen([track])
        assert screen._downmix[0] == DownmixMode.DOWN6

    def test_suspicious_not_auto_preselected(self) -> None:
        track = _t(channels=6)
        track.audio_profile = _suspicious_profile()
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_real_not_auto_preselected(self) -> None:
        track = _t(channels=6)
        track.audio_profile = _real_profile()
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_no_profile_not_auto_preselected(self) -> None:
        track = _t(channels=6)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_fake_without_suggested_not_auto_preselected(self) -> None:
        track = _t(channels=6)
        track.audio_profile = _fake_profile(None)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_fake_stereo_on_stereo_track_skipped_by_channel_guard(self) -> None:
        """FAKE + suggested STEREO on a 2-channel track: guard skips preselect."""
        track = _t(channels=2, layout="stereo")
        track.audio_profile = _fake_profile(DownmixMode.STEREO)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_fake_mono_on_mono_track_skipped_by_channel_guard(self) -> None:
        """FAKE + suggested MONO on a 1-channel track: guard skips preselect."""
        track = _t(channels=1, layout="mono")
        track.audio_profile = _fake_profile(DownmixMode.MONO)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_fake_down6_on_5_1_skipped_by_channel_guard(self) -> None:
        """FAKE + suggested DOWN6 on a 5.1 track: guard skips preselect."""
        track = _t(channels=6)
        track.audio_profile = _fake_profile(DownmixMode.DOWN6)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_fake_with_unknown_channels_skipped(self) -> None:
        track = _t(channels=None)
        track.audio_profile = _fake_profile(DownmixMode.STEREO)
        screen = self.make_screen([track])
        assert screen._downmix[0] is None

    def test_subtitle_track_type_skips_auto_preselect(self) -> None:
        """Non-AUDIO track_type never auto-preselects (and never touches _downmix)."""
        sub = make_track(
            index=2,
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            codec_id=None,
            channels=None,
            channel_layout=None,
            bitrate=None,
        )
        # Even if somehow a subtitle had an audio_profile (shouldn't happen),
        # track_type=SUBTITLE must bypass the preselect loop entirely.
        movie = make_movie(
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                duration_s=120.0, bitrate=10_000_000,
            ),
            subtitle_tracks=[sub],
        )
        screen = TrackSelectorScreen(movie, [sub], TrackType.SUBTITLE)
        assert screen._downmix == [None]


# ---------------------------------------------------------------------------
# Detector detail panel (Task 17)
# ---------------------------------------------------------------------------


class TestBarAndWord:
    def test_silent_floor(self) -> None:
        from furnace.ui.tui import _bar_and_word
        bar, word = _bar_and_word(-80.0)
        assert word == "silent"
        assert bar.count("#") == 0

    def test_very_quiet(self) -> None:
        from furnace.ui.tui import _bar_and_word
        _, word = _bar_and_word(-58.0)
        assert word == "very quiet"

    def test_quiet(self) -> None:
        from furnace.ui.tui import _bar_and_word
        _, word = _bar_and_word(-45.0)
        assert word == "quiet"

    def test_loud(self) -> None:
        from furnace.ui.tui import _bar_and_word
        _, word = _bar_and_word(-30.0)
        assert word == "loud"

    def test_full_clamps_at_zero(self) -> None:
        from furnace.ui.tui import _bar_and_word
        bar, word = _bar_and_word(5.0)
        assert word == "full"
        assert bar.count("#") == 20


class TestRenderDetectorPanel:
    def test_none_track(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        assert "---" in _render_detector_panel(None)

    def test_subtitle_track_bypass(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        sub = make_track(
            index=0, track_type=TrackType.SUBTITLE, codec_name="subrip", codec_id=None,
            channels=None, channel_layout=None, bitrate=None,
        )
        assert "---" in _render_detector_panel(sub)

    def test_not_analyzed(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        assert "not analyzed" in _render_detector_panel(track)

    def test_real_surround(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        track.audio_profile = _real_profile()
        panel = _render_detector_panel(track)
        assert "real surround" in panel
        assert "   L    [" in panel
        assert "   C    [" in panel

    def test_real_stereo(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t(channels=2, layout="stereo")
        track.audio_profile = AudioProfile(
            verdict=Verdict.REAL, score=0, suggested=None, reasons=(),
            metrics=AudioMetrics(
                channels=2, rms_l=-20.0, rms_r=-20.5,
                rms_c=None, rms_lfe=None, rms_ls=None, rms_rs=None,
                rms_lb=None, rms_rb=None,
                corr_lr=0.3, corr_ls_l=None, corr_rs_r=None,
                corr_ls_rs=None, corr_lb_ls=None, corr_rb_rs=None,
            ),
        )
        panel = _render_detector_panel(track)
        assert "real stereo" in panel

    def test_fake_surround_with_annotations(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        track.audio_profile = AudioProfile(
            verdict=Verdict.FAKE, score=3, suggested=DownmixMode.STEREO,
            reasons=(
                "both surrounds are silent (Ls=-55, Rs=-55 dB)",
                "LFE is dead (-80 dB)",
                "center is way louder than everything else (15 dB above)",
            ),
            metrics=AudioMetrics(
                channels=6,
                rms_l=-40.0, rms_r=-40.0, rms_c=-25.0, rms_lfe=-80.0,
                rms_ls=-55.0, rms_rs=-55.0, rms_lb=None, rms_rb=None,
                corr_lr=0.5, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.2,
                corr_lb_ls=None, corr_rb_rs=None,
            ),
        )
        panel = _render_detector_panel(track)
        assert "FAKE surround" in panel
        assert "suggested STEREO" in panel
        assert "<- silent" in panel
        assert "<- dead" in panel
        assert "<- dominant" in panel

    def test_fake_stereo(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t(channels=2, layout="stereo")
        track.audio_profile = AudioProfile(
            verdict=Verdict.FAKE, score=2, suggested=DownmixMode.MONO,
            reasons=("left and right are identical (mono) - corr=0.999, diff=0.0 dB",),
            metrics=AudioMetrics(
                channels=2, rms_l=-22.0, rms_r=-22.0,
                rms_c=None, rms_lfe=None, rms_ls=None, rms_rs=None,
                rms_lb=None, rms_rb=None,
                corr_lr=1.0, corr_ls_l=None, corr_rs_r=None,
                corr_ls_rs=None, corr_lb_ls=None, corr_rb_rs=None,
            ),
        )
        panel = _render_detector_panel(track)
        assert "FAKE stereo" in panel
        assert "suggested MONO" in panel
        assert "<- mono" in panel

    def test_suspicious(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        track.audio_profile = _suspicious_profile()
        panel = _render_detector_panel(track)
        assert "SUSPICIOUS" in panel
        assert "<- identical" in panel

    def test_suspicious_no_suggested(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        track.audio_profile = AudioProfile(
            verdict=Verdict.SUSPICIOUS, score=1, suggested=None,
            reasons=("something fishy",),
            metrics=AudioMetrics(
                channels=6,
                rms_l=-22.0, rms_r=-22.0, rms_c=-18.0, rms_lfe=-25.0,
                rms_ls=-25.0, rms_rs=-25.0, rms_lb=None, rms_rb=None,
                corr_lr=0.3, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.2,
                corr_lb_ls=None, corr_rb_rs=None,
            ),
        )
        panel = _render_detector_panel(track)
        assert "(none)" in panel

    def test_7_1_with_back_surround_annotations(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t(channels=8, layout="7.1")
        track.audio_profile = AudioProfile(
            verdict=Verdict.FAKE, score=3, suggested=DownmixMode.STEREO,
            reasons=("surrounds are a copy of fronts",),
            metrics=AudioMetrics(
                channels=8,
                rms_l=-20.0, rms_r=-20.0, rms_c=-18.0, rms_lfe=-25.0,
                rms_ls=-22.0, rms_rs=-22.0, rms_lb=-22.0, rms_rb=-22.0,
                corr_lr=0.5, corr_ls_l=0.99, corr_rs_r=0.99, corr_ls_rs=0.3,
                corr_lb_ls=0.4, corr_rb_rs=0.4,
            ),
        )
        panel = _render_detector_panel(track)
        assert "FAKE surround" in panel
        assert "   Lb  " in panel
        assert "   Rb  " in panel
        assert "<- copy of L" in panel
        assert "<- copy of R" in panel

    def test_ls_rs_identical_annotation(self) -> None:
        from furnace.ui.tui import _render_detector_panel
        track = _t()
        track.audio_profile = AudioProfile(
            verdict=Verdict.SUSPICIOUS, score=1, suggested=DownmixMode.STEREO,
            reasons=("left and right surrounds carry the same signal",),
            metrics=AudioMetrics(
                channels=6,
                rms_l=-22.0, rms_r=-22.0, rms_c=-18.0, rms_lfe=-25.0,
                rms_ls=-25.0, rms_rs=-25.0, rms_lb=None, rms_rb=None,
                corr_lr=0.3, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.9,
                corr_lb_ls=None, corr_rb_rs=None,
            ),
        )
        panel = _render_detector_panel(track)
        assert "<- identical" in panel


class TestModeLabel:
    def test_stereo(self) -> None:
        from furnace.ui.tui import _mode_label
        assert _mode_label(DownmixMode.STEREO) == "STEREO"

    def test_mono(self) -> None:
        from furnace.ui.tui import _mode_label
        assert _mode_label(DownmixMode.MONO) == "MONO"

    def test_down6(self) -> None:
        from furnace.ui.tui import _mode_label
        assert _mode_label(DownmixMode.DOWN6) == "DOWN6"

    def test_none(self) -> None:
        from furnace.ui.tui import _mode_label
        assert _mode_label(None) == "(none)"


class TestRefreshDetectorPanelOnSubtitle:
    def test_subtitle_track_type_short_circuits(self) -> None:
        """_refresh_detector_panel returns early for non-audio screens."""
        sub = make_track(
            index=2, track_type=TrackType.SUBTITLE, codec_name="subrip", codec_id=None,
            channels=None, channel_layout=None, bitrate=None,
        )
        movie = make_movie(
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                duration_s=120.0, bitrate=10_000_000,
            ),
            subtitle_tracks=[sub],
        )
        screen = TrackSelectorScreen(movie, [sub], TrackType.SUBTITLE)
        # No panel widget exists for subtitle screens; method must not raise.
        screen._refresh_detector_panel()
