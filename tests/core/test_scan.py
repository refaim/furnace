from __future__ import annotations

import pytest

from furnace.core.scan import (
    AudioTrackSummary,
    ScanRow,
    SubtitleTrackSummary,
    parse_furnace_version,
    parse_version_arg,
    row_matches,
    summarize_streams,
)

# ---------------------------------------------------------------------------
# parse_furnace_version
# ---------------------------------------------------------------------------


def test_parse_furnace_version_valid_tag() -> None:
    assert parse_furnace_version("Furnace v1.19.3") == (1, 19, 3)


def test_parse_furnace_version_multi_digit() -> None:
    assert parse_furnace_version("Furnace v12.345.6789") == (12, 345, 6789)


def test_parse_furnace_version_none() -> None:
    assert parse_furnace_version(None) is None


def test_parse_furnace_version_foreign_tag() -> None:
    assert parse_furnace_version("Lavf60.16.100") is None


def test_parse_furnace_version_malformed_missing_patch() -> None:
    assert parse_furnace_version("Furnace v1.19") is None


def test_parse_furnace_version_malformed_no_v() -> None:
    assert parse_furnace_version("Furnace 1.19.3") is None


def test_parse_furnace_version_malformed_trailing() -> None:
    assert parse_furnace_version("Furnace v1.19.3-dirty") is None


def test_parse_furnace_version_malformed_leading() -> None:
    assert parse_furnace_version("x Furnace v1.19.3") is None


def test_parse_furnace_version_empty() -> None:
    assert parse_furnace_version("") is None


# ---------------------------------------------------------------------------
# parse_version_arg
# ---------------------------------------------------------------------------


def test_parse_version_arg_valid() -> None:
    assert parse_version_arg("1.19.3") == (1, 19, 3)


def test_parse_version_arg_multi_digit() -> None:
    assert parse_version_arg("10.0.255") == (10, 0, 255)


@pytest.mark.parametrize(
    "bad",
    ["1.19", "1", "1.19.3.4", "v1.19.3", "1.19.x", "", "1.19.", "a.b.c"],
)
def test_parse_version_arg_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid version"):
        parse_version_arg(bad)


# ---------------------------------------------------------------------------
# summarize_streams
# ---------------------------------------------------------------------------


def test_summarize_streams_multi_audio_and_subs() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "hevc"},
            {
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 6,
                "tags": {"language": "rus"},
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "tags": {"language": "eng"},
            },
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {"language": "eng"},
            },
            {
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "rus"},
            },
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == "hevc"
    assert audio == (
        AudioTrackSummary(language="rus", codec="ac3", channels=6),
        AudioTrackSummary(language="eng", codec="aac", channels=2),
    )
    assert subs == (
        SubtitleTrackSummary(language="eng", codec="subrip"),
        SubtitleTrackSummary(language="rus", codec="hdmv_pgs_subtitle"),
    )


def test_summarize_streams_missing_language_is_none() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "ac3", "channels": 2},
            {"codec_type": "subtitle", "codec_name": "subrip"},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video is None
    assert audio == (AudioTrackSummary(language=None, codec="ac3", channels=2),)
    assert subs == (SubtitleTrackSummary(language=None, codec="subrip"),)


def test_summarize_streams_no_video_stream() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "flac", "channels": 8, "tags": {"language": "jpn"}},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video is None
    assert audio == (AudioTrackSummary(language="jpn", codec="flac", channels=8),)
    assert subs == ()


def test_summarize_streams_missing_channels_is_none() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "ac3", "tags": {"language": "eng"}},
        ]
    }
    _, audio, _ = summarize_streams(probe)
    assert audio == (AudioTrackSummary(language="eng", codec="ac3", channels=None),)


def test_summarize_streams_missing_codec_name_is_unknown() -> None:
    probe = {
        "streams": [
            {"codec_type": "video"},
            {"codec_type": "audio", "tags": {"language": "eng"}},
            {"codec_type": "subtitle", "tags": {"language": "eng"}},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == "unknown"
    assert audio == (AudioTrackSummary(language="eng", codec="unknown", channels=None),)
    assert subs == (SubtitleTrackSummary(language="eng", codec="unknown"),)


def test_summarize_streams_empty_streams() -> None:
    assert summarize_streams({"streams": []}) == (None, (), ())


def test_summarize_streams_missing_streams_key() -> None:
    assert summarize_streams({}) == (None, (), ())


def test_summarize_streams_first_video_wins() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "mpeg2video"},
            {"codec_type": "video", "codec_name": "hevc"},
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video == "mpeg2video"


# ---------------------------------------------------------------------------
# ScanRow
# ---------------------------------------------------------------------------


def test_scan_row_defaults_unreadable_false() -> None:
    from pathlib import Path

    row = ScanRow(
        path=Path("a.mkv"),
        furnace_version=(1, 19, 3),
        video_codec="hevc",
        audio=(),
        subtitles=(),
    )
    assert row.unreadable is False
    assert row.furnace_version == (1, 19, 3)


def test_scan_row_unreadable() -> None:
    from pathlib import Path

    row = ScanRow(
        path=Path("bad.mkv"),
        furnace_version=None,
        video_codec=None,
        audio=(),
        subtitles=(),
        unreadable=True,
    )
    assert row.unreadable is True


# ---------------------------------------------------------------------------
# row_matches
# ---------------------------------------------------------------------------


def test_row_matches_no_predicate_encoded() -> None:
    assert row_matches((1, 19, 3), not_encoded=False, encoded=False, max_version=None) is True


def test_row_matches_no_predicate_not_encoded() -> None:
    assert row_matches(None, not_encoded=False, encoded=False, max_version=None) is True


def test_row_matches_not_encoded_true_when_none() -> None:
    assert row_matches(None, not_encoded=True, encoded=False, max_version=None) is True


def test_row_matches_not_encoded_false_when_version() -> None:
    assert row_matches((1, 0, 0), not_encoded=True, encoded=False, max_version=None) is False


def test_row_matches_encoded_true_when_version() -> None:
    assert row_matches((1, 0, 0), not_encoded=False, encoded=True, max_version=None) is True


def test_row_matches_encoded_false_when_none() -> None:
    assert row_matches(None, not_encoded=False, encoded=True, max_version=None) is False


def test_row_matches_max_version_below() -> None:
    assert row_matches((1, 19, 2), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_max_version_boundary_equal() -> None:
    assert row_matches((1, 19, 3), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_max_version_above() -> None:
    assert row_matches((1, 19, 4), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is False


def test_row_matches_max_version_none_version() -> None:
    # A not-encoded file cannot satisfy a --max-version predicate.
    assert row_matches(None, not_encoded=False, encoded=False, max_version=(1, 19, 3)) is False


def test_row_matches_union_not_encoded_or_max_version() -> None:
    # not_encoded matches a None version even though max_version would not.
    assert row_matches(None, not_encoded=True, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_union_encoded_or_max_version() -> None:
    # encoded matches any version even though max_version would not (too new).
    assert row_matches((2, 0, 0), not_encoded=False, encoded=True, max_version=(1, 19, 3)) is True


def test_row_matches_union_all_false() -> None:
    assert row_matches((2, 0, 0), not_encoded=True, encoded=False, max_version=(1, 19, 3)) is False
