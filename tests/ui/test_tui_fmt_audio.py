"""Pure-function tests for `_fmt_audio_track`."""
from __future__ import annotations

from pathlib import Path

from furnace.core.models import AudioCodecId, DownmixMode, Track, TrackType
from furnace.ui.tui import TrackSelection, TrackSelectorScreen, _fmt_audio_track
from tests.conftest import make_movie, make_track, make_video_info


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
