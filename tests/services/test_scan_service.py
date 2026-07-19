from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from furnace.core.outdated import EncoderFamily, Fix, Severity
from furnace.core.scan import AudioTrackSummary, ScanRow, SubtitleTrackSummary, VideoSummary
from furnace.services.scan_service import ScanService


def make_probe(
    *,
    encoder: str | None = None,
    encoder_settings: str | None = None,
    video: str | None = "hevc",
    pix_fmt: str | None = None,
    color_transfer: str | None = None,
    color_space: str | None = None,
    height: int | None = None,
    audios: tuple[tuple[str | None, str, int | None], ...] = (),
    subs: tuple[tuple[str | None, str], ...] = (),
) -> dict[str, Any]:
    streams: list[dict[str, Any]] = []
    if video is not None:
        vs: dict[str, Any] = {"codec_type": "video", "codec_name": video}
        if pix_fmt is not None:
            vs["pix_fmt"] = pix_fmt
        if color_transfer is not None:
            vs["color_transfer"] = color_transfer
        if color_space is not None:
            vs["color_space"] = color_space
        if height is not None:
            vs["height"] = height
        streams.append(vs)
    for lang, codec, channels in audios:
        s: dict[str, Any] = {"codec_type": "audio", "codec_name": codec, "channels": channels}
        if lang is not None:
            s["tags"] = {"language": lang}
        streams.append(s)
    for lang, codec in subs:
        sub: dict[str, Any] = {"codec_type": "subtitle", "codec_name": codec}
        if lang is not None:
            sub["tags"] = {"language": lang}
        streams.append(sub)
    tags: dict[str, Any] = {}
    if encoder is not None:
        tags["ENCODER"] = encoder
    if encoder_settings is not None:
        tags["ENCODER_SETTINGS"] = encoder_settings
    fmt: dict[str, Any] = {"tags": tags} if tags else {}
    return {"streams": streams, "format": fmt}


def make_service(probe_map: dict[Path, dict[str, Any]] | None = None) -> tuple[ScanService, MagicMock]:
    prober = MagicMock()
    if probe_map is not None:
        prober.probe.side_effect = lambda p: probe_map[p]
    return ScanService(prober=prober), prober


class TestRowBuilding:
    def test_builds_full_row(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = make_probe(
            encoder="Furnace v1.19.3",
            video="hevc",
            pix_fmt="yuv420p10le",
            color_transfer="smpte2084",
            audios=(("rus", "eac3", 6), (None, "aac", 2)),
            subs=(("eng", "subrip"), (None, "hdmv_pgs_subtitle")),
        )
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(movie)

        assert rows == [
            ScanRow(
                path=movie,
                furnace_version=(1, 19, 3),
                video=VideoSummary(codec="hevc", bit_depth=10, hdr="HDR10", color_transfer="smpte2084"),
                audio=(
                    AudioTrackSummary(language="rus", codec="eac3", channels=6),
                    AudioTrackSummary(language=None, codec="aac", channels=2),
                ),
                subtitles=(
                    SubtitleTrackSummary(language="eng", codec="subrip"),
                    SubtitleTrackSummary(language=None, codec="hdmv_pgs_subtitle"),
                ),
            )
        ]

    def test_not_encoded_file_has_no_version(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = make_probe(encoder="Lavf60.16.100", video="h264")
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(movie)

        assert rows[0].furnace_version is None
        assert rows[0].video.codec == "h264"

    def test_lowercase_encoder_tag_is_parsed(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = {
            "streams": [{"codec_type": "video", "codec_name": "hevc"}],
            "format": {"tags": {"encoder": "Furnace v1.19.3"}},
        }
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(movie)

        assert rows[0].furnace_version == (1, 19, 3)

    def test_uppercase_encoder_takes_precedence_over_lowercase(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = {
            "streams": [{"codec_type": "video", "codec_name": "hevc"}],
            "format": {"tags": {"ENCODER": "Furnace v1.19.3", "encoder": "Lavf60"}},
        }
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(movie)

        assert rows[0].furnace_version == (1, 19, 3)

    def test_no_video_stream_yields_none_codec(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = make_probe(video=None, audios=(("eng", "aac", 2),))
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(movie)

        assert rows[0].video == VideoSummary(None, None, None)


class TestDiscovery:
    def test_single_file_root_yields_one_entry(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        service, _ = make_service({movie: make_probe()})

        rows, _ = service.scan(movie)

        assert [r.path for r in rows] == [movie]

    def test_single_non_video_file_yields_empty(self, tmp_path: Path) -> None:
        txt = tmp_path / "readme.txt"
        txt.touch()
        service, prober = make_service()

        rows, _ = service.scan(txt)

        assert rows == []
        prober.probe.assert_not_called()

    def test_recursion_finds_nested_files_sorted(self, tmp_path: Path) -> None:
        sub = tmp_path / "Action"
        sub.mkdir()
        a = sub / "a.mkv"
        b = tmp_path / "b.mp4"
        a.touch()
        b.touch()
        probe_map = {a: make_probe(), b: make_probe()}
        service, _ = make_service(probe_map)

        rows, _ = service.scan(tmp_path)

        assert [r.path for r in rows] == [a, b]

    def test_non_video_files_skipped_in_walk(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "movie.nfo").touch()
        service, _ = make_service({movie: make_probe()})

        rows, _ = service.scan(tmp_path)

        assert [r.path for r in rows] == [movie]

    def test_directory_entries_skipped(self, tmp_path: Path) -> None:
        weird_dir = tmp_path / "season.mkv"
        weird_dir.mkdir()
        movie = tmp_path / "movie.mkv"
        movie.touch()
        service, _ = make_service({movie: make_probe()})

        rows, _ = service.scan(tmp_path)

        assert [r.path for r in rows] == [movie]

    def test_furnace_demux_dir_skipped(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        demux = tmp_path / ".furnace_demux"
        demux.mkdir()
        (demux / "intermediate.mkv").touch()
        service, _ = make_service({movie: make_probe()})

        rows, _ = service.scan(tmp_path)

        assert [r.path for r in rows] == [movie]


class TestFiltering:
    def _three_file_service(self, tmp_path: Path) -> ScanService:
        plain = tmp_path / "a_plain.mkv"
        old = tmp_path / "b_old.mkv"
        new = tmp_path / "c_new.mkv"
        for p in (plain, old, new):
            p.touch()
        probe_map = {
            plain: make_probe(encoder=None),
            old: make_probe(encoder="Furnace v1.10.0"),
            new: make_probe(encoder="Furnace v1.20.0"),
        }
        service, _ = make_service(probe_map)
        return service

    def test_no_filter_shows_all(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, _ = service.scan(tmp_path)
        assert [r.path.name for r in rows] == ["a_plain.mkv", "b_old.mkv", "c_new.mkv"]

    def test_not_encoded_filter(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, _ = service.scan(tmp_path, not_encoded=True)
        assert [r.path.name for r in rows] == ["a_plain.mkv"]

    def test_encoded_filter(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, _ = service.scan(tmp_path, encoded=True)
        assert [r.path.name for r in rows] == ["b_old.mkv", "c_new.mkv"]

    def test_max_version_filter(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, _ = service.scan(tmp_path, max_version=(1, 19, 3))
        assert [r.path.name for r in rows] == ["b_old.mkv"]

    def test_union_of_predicates(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, _ = service.scan(tmp_path, not_encoded=True, max_version=(1, 19, 3))
        assert [r.path.name for r in rows] == ["a_plain.mkv", "b_old.mkv"]

    def test_total_counts_all_files_regardless_of_filter(self, tmp_path: Path) -> None:
        service = self._three_file_service(tmp_path)
        rows, total = service.scan(tmp_path, not_encoded=True)
        assert len(rows) == 1
        assert total == 3

    def test_total_includes_unreadable_files(self, tmp_path: Path) -> None:
        good = tmp_path / "a_good.mkv"
        bad = tmp_path / "b_bad.mkv"
        good.touch()
        bad.touch()
        prober = MagicMock()

        def probe(p: Path) -> dict[str, Any]:
            if p == bad:
                raise OSError("boom")
            return make_probe()

        prober.probe.side_effect = probe
        service = ScanService(prober=prober)

        rows, total = service.scan(tmp_path)

        assert total == 2
        assert len(rows) == 2


class TestUnreadable:
    @pytest.mark.parametrize("exc", [OSError("boom"), RuntimeError("boom"), ValueError("boom")])
    def test_probe_failure_yields_unreadable_row(self, tmp_path: Path, exc: Exception) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        service, prober = make_service()
        prober.probe.side_effect = exc

        rows, _ = service.scan(movie)

        assert rows == [
            ScanRow(
                path=movie,
                furnace_version=None,
                video=VideoSummary(None, None, None),
                audio=(),
                subtitles=(),
                unreadable=True,
            )
        ]

    def test_unreadable_row_never_dropped_by_filter(self, tmp_path: Path) -> None:
        good = tmp_path / "a_good.mkv"
        bad = tmp_path / "b_bad.mkv"
        good.touch()
        bad.touch()
        prober = MagicMock()

        def probe(p: Path) -> dict[str, Any]:
            if p == bad:
                raise OSError("boom")
            return make_probe(encoder="Furnace v1.10.0")

        prober.probe.side_effect = probe
        service = ScanService(prober=prober)

        rows, _ = service.scan(tmp_path, encoded=True)

        assert [r.path.name for r in rows] == ["a_good.mkv", "b_bad.mkv"]
        assert rows[1].unreadable is True


class TestEncoderFamily:
    def test_reads_encoder_settings_family(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = make_probe(
            encoder="Furnace v2.1.0",
            encoder_settings="av1_nvenc / NVEncC=8.00 / main",
            video="av1",
        )
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert rows[0].encoder_family is EncoderFamily.AV1_NVENC

    def test_encoder_settings_lowercase_fallback(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = {
            "streams": [{"codec_type": "video", "codec_name": "av1", "height": 480}],
            "format": {
                "tags": {
                    "ENCODER": "Furnace v2.0.5",
                    "encoder_settings": "av1_svt / SVT-AV1 / preset=4",
                }
            },
        }
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert rows[0].encoder_family is EncoderFamily.AV1_SVT

    def test_uppercase_encoder_settings_precedence(self, tmp_path: Path) -> None:
        movie = tmp_path / "movie.mkv"
        movie.touch()
        probe = {
            "streams": [{"codec_type": "video", "codec_name": "av1", "height": 2160}],
            "format": {
                "tags": {
                    "ENCODER": "Furnace v2.1.0",
                    "ENCODER_SETTINGS": "av1_nvenc / main",
                    "encoder_settings": "av1_svt / SVT-AV1",
                }
            },
        }
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert rows[0].encoder_family is EncoderFamily.AV1_NVENC


class TestOutdated:
    def test_keeps_only_flagged_files(self, tmp_path: Path) -> None:
        clean = tmp_path / "a_clean.mkv"
        foreign = tmp_path / "b_foreign.mkv"
        old = tmp_path / "c_old.mkv"
        for p in (clean, foreign, old):
            p.touch()
        probe_map = {
            clean: make_probe(
                encoder="Furnace v2.9.0",
                encoder_settings="av1_nvenc / main",
                video="av1",
                height=1080,
                color_space="bt709",
                audios=(("eng", "aac", 2),),
            ),
            foreign: make_probe(encoder="Lavf60", video="h264", height=1080, color_space="bt709"),
            old: make_probe(
                encoder="Furnace v2.1.0",
                encoder_settings="av1_nvenc / main",
                video="av1",
                height=1080,
                color_space="bt709",
            ),
        }
        service, _ = make_service(probe_map)

        rows, total = service.scan(tmp_path, outdated=True)

        assert [r.path.name for r in rows] == ["b_foreign.mkv", "c_old.mkv"]
        assert total == 3

    def test_attaches_defects_to_row(self, tmp_path: Path) -> None:
        foreign = tmp_path / "f.mkv"
        foreign.touch()
        probe = make_probe(encoder=None, video="h264", height=1080, color_space="bt709")
        service, _ = make_service({foreign: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert len(rows[0].defects) == 1
        defect = rows[0].defects[0]
        assert defect.reason == "h264"
        assert defect.severity is Severity.FOREIGN
        assert defect.fix is Fix.RE_ENCODE

    def test_unreadable_row_always_kept_and_classified(self, tmp_path: Path) -> None:
        good = tmp_path / "a_good.mkv"
        bad = tmp_path / "b_bad.mkv"
        good.touch()
        bad.touch()
        prober = MagicMock()

        def probe(p: Path) -> dict[str, Any]:
            if p == bad:
                raise OSError("boom")
            return make_probe(
                encoder="Furnace v2.9.0",
                encoder_settings="av1_nvenc / main",
                video="av1",
                height=1080,
                color_space="bt709",
                audios=(("eng", "aac", 2),),
            )

        prober.probe.side_effect = probe
        service = ScanService(prober=prober)

        rows, total = service.scan(tmp_path, outdated=True)

        assert [r.path.name for r in rows] == ["b_bad.mkv"]
        assert rows[0].unreadable is True
        assert rows[0].defects[0].reason == "unreadable"
        assert total == 2

    def test_stacked_defects_sorted_worst_first(self, tmp_path: Path) -> None:
        movie = tmp_path / "old.mkv"
        movie.touch()
        probe = make_probe(
            encoder="Furnace v2.0.5",
            encoder_settings="av1_nvenc / main",
            video="av1",
            height=480,
            color_space=None,
        )
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        reasons = [d.reason for d in rows[0].defects]
        assert reasons == ["fps drift", "crop 4px", "grain loss", "color tags"]

    def test_pre_target_quality_hdr_flagged_soft_qvbr(self, tmp_path: Path) -> None:
        movie = tmp_path / "xmen.mkv"
        movie.touch()
        probe = make_probe(
            encoder="Furnace v2.9.0",
            encoder_settings="av1_nvenc / NVEncC=8.00 / main",
            video="av1",
            height=1600,
            color_space="bt2020nc",
            color_transfer="smpte2084",
        )
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert [d.reason for d in rows[0].defects] == ["soft QVBR (HDR)"]
        assert rows[0].defects[0].fix is Fix.RE_ENCODE

    def test_current_hdr_encode_not_flagged(self, tmp_path: Path) -> None:
        movie = tmp_path / "fresh.mkv"
        movie.touch()
        probe = make_probe(
            encoder="Furnace v2.19.1",
            encoder_settings="av1_nvenc / NVEncC=8.00 / main",
            video="av1",
            height=1600,
            color_space="bt2020nc",
            color_transfer="smpte2084",
        )
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path, outdated=True)

        assert rows == []

    def test_normal_mode_does_not_attach_defects(self, tmp_path: Path) -> None:
        movie = tmp_path / "old.mkv"
        movie.touch()
        probe = make_probe(encoder="Furnace v2.0.5", encoder_settings="av1_nvenc / main", video="av1", height=480)
        service, _ = make_service({movie: probe})

        rows, _ = service.scan(tmp_path)

        assert rows[0].defects == ()
