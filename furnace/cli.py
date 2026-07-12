from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Header

from .adapters.dovi_tool import DoviToolAdapter
from .adapters.eac3to import Eac3toAdapter
from .adapters.ffmpeg import FFmpegAdapter
from .adapters.makemkv import MakemkvAdapter
from .adapters.mkclean import MkcleanAdapter
from .adapters.mkvmerge import MkvmergeAdapter
from .adapters.mkvpropedit import MkvpropeditAdapter
from .adapters.mpv import MpvAdapter
from .adapters.nvencc import NVEncCAdapter
from .adapters.qaac import QaacAdapter
from .adapters.svtav1 import SvtAv1Adapter
from .adapters.vship_metrics import VshipMetricsAdapter
from .config import load_config
from .core.detect import classify_grain, needs_grain_probe
from .core.models import (
    DiscSource,
    DiscTitle,
    DiscType,
    DownmixMode,
    JobStatus,
    Movie,
    Plan,
    ScanResult,
    Track,
    TrackType,
)
from .core.scan import parse_version_arg
from .plan import load_plan, save_plan
from .services.analysis_pipeline import AnalysisPipeline
from .services.analyzer import Analyzer
from .services.disc_demuxer import DiscDemuxer
from .services.executor import Executor
from .services.planner import PlannerService
from .services.scan_service import ScanService
from .services.scanner import Scanner
from .ui.plan_console import RichPlanReporter
from .ui.progress import ReportPrinter
from .ui.run_tui import RunApp
from .ui.scan_table import render_scan_table
from .ui.tui import (
    FileSelection,
    FileSelectorScreen,
    LanguageSelectorScreen,
    PlaylistSelectorScreen,
    TrackSelection,
    TrackSelectorScreen,
)

app = typer.Typer()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Textual app runners
# ---------------------------------------------------------------------------

def _run_screen_app[T](screen_factory: Callable[[], Screen[T]]) -> T | None:
    """Build a minimal Textual App that pushes a single Screen and returns its dismiss result.

    `screen_factory` is invoked inside `on_mount` so the caller doesn't need to
    pre-instantiate the screen before the app event loop is ready.
    """
    result_holder: list[T | None] = [None]

    class _ScreenApp(App[T]):
        TITLE = "Furnace"

        def compose(self) -> ComposeResult:
            yield Header()

        def on_mount(self) -> None:
            def _on_dismiss(result: T | None) -> None:
                result_holder[0] = result
                self.exit(result)

            self.push_screen(screen_factory(), _on_dismiss)

    _ScreenApp().run()
    return result_holder[0]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _make_preview_track_cb(movie: Movie, mpv_adapter: MpvAdapter) -> Callable[[Track], None]:
    """Create a preview callback with closure over movie and mpv adapter."""

    def _preview_track(track: Track) -> None:
        if track.track_type == TrackType.AUDIO:
            mpv_adapter.preview_audio(movie.main_file, track.source_file, track.index)
        else:
            mpv_adapter.preview_subtitle(movie.main_file, track.source_file, track.index)

    return _preview_track


# ---------------------------------------------------------------------------
# Track / language selector wrappers
# ---------------------------------------------------------------------------


def _select_tracks_tui(
    movie: Movie,
    candidates: list[Track],
    track_type: TrackType,
    mpv_adapter: MpvAdapter,
    *,
    allow_relabel: bool = False,
    lang_list: list[str] | None = None,
    app_runner: Callable[[Callable[[], Screen[TrackSelection]]], TrackSelection | None] = _run_screen_app,
) -> TrackSelection:
    """Run Textual TrackSelectorScreen synchronously for user to pick tracks."""

    def _factory() -> Screen[TrackSelection]:
        return TrackSelectorScreen(
            movie=movie,
            tracks=candidates,
            track_type=track_type,
            preview_cb=_make_preview_track_cb(movie, mpv_adapter),
            allow_relabel=allow_relabel,
            lang_list=lang_list,
        )

    result = app_runner(_factory)
    if result is None:
        return TrackSelection(tracks=[], downmix={})
    return result


def _select_tracks_tui_for_planner(
    movie: Movie,
    candidates: list[Track],
    track_type: TrackType,
    mpv_adapter: MpvAdapter,
    downmix_overrides: dict[tuple[Path, int], DownmixMode],
    lang_overrides: dict[tuple[Path, int], str],
    *,
    allow_relabel: bool = False,
    lang_list: list[str] | None = None,
    app_runner: Callable[[Callable[[], Screen[TrackSelection]]], TrackSelection | None] = _run_screen_app,
) -> list[Track]:
    """Planner-facing wrapper: returns list[Track] and mutates the shared override dicts.

    Relabel languages picked in the TUI merge into ``lang_overrides`` (all track
    types); audio downmix choices merge into ``downmix_overrides``.
    """
    result = _select_tracks_tui(
        movie,
        candidates,
        track_type,
        mpv_adapter,
        allow_relabel=allow_relabel,
        lang_list=lang_list,
        app_runner=app_runner,
    )
    lang_overrides.update(result.languages)
    if track_type == TrackType.AUDIO:
        downmix_overrides.update(result.downmix)
    return result.tracks


def _resolve_und_language_tui(
    movie: Movie,
    track: Track,
    lang_list: list[str],
    mpv_adapter: MpvAdapter,
    *,
    app_runner: Callable[[Callable[[], Screen[str]]], str | None] = _run_screen_app,
) -> str:
    """Run Textual LanguageSelectorScreen synchronously for user to pick a language."""

    def _factory() -> Screen[str]:
        return LanguageSelectorScreen(
            track=track,
            lang_list=lang_list,
            preview_cb=_make_preview_track_cb(movie, mpv_adapter),
            movie=movie,
        )

    result = app_runner(_factory)
    if result is None:
        return lang_list[0]
    return result


# ---------------------------------------------------------------------------
# Disc demux helpers
# ---------------------------------------------------------------------------


def _collect_selected_titles(
    detected_discs: list[DiscSource],
    disc_titles: dict[DiscSource, list[DiscTitle]],
    *,
    reporter: RichPlanReporter | None = None,
    playlist_app_runner: Callable[
        [Callable[[], Screen[list[DiscTitle]]]], list[DiscTitle] | None
    ] = _run_screen_app,
) -> dict[DiscSource, list[DiscTitle]]:
    """For each detected disc, pick which titles to demux from the pre-listed
    playlists. Pauses the reporter only around the interactive playlist screen."""
    selected_titles: dict[DiscSource, list[DiscTitle]] = {}
    for disc in detected_discs:
        playlists = disc_titles.get(disc, [])
        if not playlists:
            logger.warning("No playlists found for disc at %s", disc.path)
            continue
        if len(playlists) == 1:
            selected_titles[disc] = playlists
            continue
        disc_label = disc.path.parent.name

        def _factory(
            _disc_label: str = disc_label,
            _playlists: list[DiscTitle] = playlists,
        ) -> Screen[list[DiscTitle]]:
            return PlaylistSelectorScreen(disc_label=_disc_label, playlists=_playlists)

        if reporter is not None:
            reporter.pause()
        picked = playlist_app_runner(_factory)
        if reporter is not None:
            reporter.resume()
        if picked:
            selected_titles[disc] = picked
    return selected_titles


def _dvd_demuxed_paths(
    detected_discs: list[DiscSource],
    selected_titles: dict[DiscSource, list[DiscTitle]],
    demuxed_paths: list[Path],
) -> set[Path]:
    """Identify which demuxed paths came from DVD sources (by filename prefix)."""
    dvd_demuxed: set[Path] = set()
    for disc in detected_discs:
        if disc.disc_type == DiscType.DVD and disc in selected_titles:
            disc_label = disc.path.parent.name
            for p in demuxed_paths:
                if p.name.startswith(disc_label):
                    dvd_demuxed.add(p)
    return dvd_demuxed


def _probe_file_infos(
    demuxed_paths: list[Path], ffmpeg_adapter: FFmpegAdapter
) -> list[tuple[Path, float, int, int]]:
    """Probe each file for duration/size/height for the file-selector UI.

    Height is read from the first video stream (0 when none is present) and
    drives the SD grain gate; duration and size feed the on-screen file list.
    Duration mirrors the analyzer's precedence exactly — the first video
    stream's ``duration`` first, falling back to ``format.duration`` — so the
    grain pre-probe seeks the same window as the headless grain verdict and the
    two agree at the classify boundary.
    """
    file_infos: list[tuple[Path, float, int, int]] = []
    for mkv_path in demuxed_paths:
        probe_data = ffmpeg_adapter.probe(mkv_path)
        fmt = probe_data.get("format", {})
        size_bytes = int(fmt.get("size", 0))
        streams = probe_data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        video_stream = video_streams[0] if video_streams else {}
        height = int(video_stream.get("height", 0))
        duration_s = 0.0
        if "duration" in video_stream:
            with contextlib.suppress(ValueError, TypeError):
                duration_s = float(video_stream["duration"])
        if duration_s == 0.0 and "duration" in fmt:
            with contextlib.suppress(ValueError, TypeError):
                duration_s = float(fmt["duration"])
        file_infos.append((mkv_path, duration_s, size_bytes, height))
    return file_infos


def _sd_grain_files(file_infos: list[tuple[Path, float, int, int]]) -> set[Path]:
    """SD files eligible for a grain toggle in the file selector.

    A height of 0 (unreadable / no video stream) is treated as non-SD so a file
    whose resolution we could not measure never triggers the fragile grain probe.
    """
    return {p for (p, _dur, _size, height) in file_infos if height > 0 and needs_grain_probe(height)}


def _classify_one(path: Path, dur: float, ffmpeg_adapter: FFmpegAdapter) -> bool:
    """Grain verdict for a single SD file, fail-soft to GRAINY.

    Mirrors the analyzer's grain stage: ``sample_grain`` returns ``[]`` (never
    raises) for expected per-window failures, but a catastrophic raise — a
    broken ffmpeg subprocess (``OSError``) or an internal parse error — is
    caught here and treated as GRAINY rather than crashing the whole ``plan``
    run. Wrongly-on costs a few extra bytes; wrongly-off smears real film grain.
    """
    try:
        return classify_grain(ffmpeg_adapter.sample_grain(path, dur))
    except (OSError, RuntimeError, ValueError):
        logger.warning("grain pre-probe failed for %s, defaulting to GRAINY", path)
        return True


def _grain_pre_probe(
    file_infos: list[tuple[Path, float, int, int]],
    sd_files: set[Path],
    ffmpeg_adapter: FFmpegAdapter,
) -> set[Path]:
    """Seed the file-selector grain default: an SD file whose sampled flicker
    classifies GRAINY starts with grain ON.

    Only ``sd_files`` are probed, so the result is always a subset of
    ``sd_files`` (never seed a default for a non-SD path). Each file is
    classified independently and fail-soft (see ``_classify_one``), so one
    file's hard probe failure never loses the others.
    """
    grain_defaults: set[Path] = set()
    for (p, dur, _size, _height) in file_infos:
        if p in sd_files and _classify_one(p, dur, ffmpeg_adapter):
            grain_defaults.add(p)
    return grain_defaults


def _run_file_selector(
    *,
    file_infos: list[tuple[Path, float, int, int]],
    dvd_files: set[Path],
    sd_files: set[Path],
    ffmpeg_adapter: FFmpegAdapter,
    mpv_adapter: MpvAdapter,
    reporter: RichPlanReporter | None,
    file_app_runner: Callable[[Callable[[], Screen[FileSelection]]], FileSelection | None],
) -> FileSelection | None:
    """Pre-probe grain for the SD files, then run the file selector.

    Brackets the interactive screen (and the slow grain pre-probe) with the
    reporter's pause/resume. Returns the ``FileSelection`` or ``None`` if the
    screen was dismissed.
    """
    if reporter is not None:
        reporter.pause()
    grain_defaults = _grain_pre_probe(file_infos, sd_files, ffmpeg_adapter)
    files_for_screen = [(p, dur, size) for (p, dur, size, _height) in file_infos]

    def _factory(
        _files: list[tuple[Path, float, int]] = files_for_screen,
        _dvd: set[Path] = dvd_files,
        _sd: set[Path] = sd_files,
        _grain: set[Path] = grain_defaults,
    ) -> Screen[FileSelection]:
        return FileSelectorScreen(
            files=_files,
            dvd_files=_dvd,
            sd_files=_sd,
            grain_defaults=_grain,
            preview_cb=lambda p, a: mpv_adapter.preview_file(p, aspect_override=a),
        )

    result = file_app_runner(_factory)
    if reporter is not None:
        reporter.resume()
    return result


def _run_disc_demux_interactive(
    *,
    source: Path,
    detected_discs: list[DiscSource],
    disc_titles: dict[DiscSource, list[DiscTitle]],
    disc_demuxer: DiscDemuxer,
    ffmpeg_adapter: FFmpegAdapter,
    mpv_adapter: MpvAdapter,
    reporter: RichPlanReporter | None = None,
    playlist_app_runner: Callable[
        [Callable[[], Screen[list[DiscTitle]]]], list[DiscTitle] | None
    ] = _run_screen_app,
    file_app_runner: Callable[
        [Callable[[], Screen[FileSelection]]], FileSelection | None
    ] = _run_screen_app,
) -> tuple[Path | None, list[Path], set[Path], dict[Path, bool]]:
    """Coordinate the interactive disc demux flow.

    Returns `(demux_dir, demuxed_paths, sar_override_paths, grain_overrides)`.
    When no discs are provided, returns `(None, [], set(), {})`.
    """
    if not detected_discs:
        return None, [], set(), {}

    selected_titles = _collect_selected_titles(
        detected_discs,
        disc_titles,
        reporter=reporter,
        playlist_app_runner=playlist_app_runner,
    )

    if not selected_titles:
        return None, [], set(), {}

    demux_dir = source / ".furnace_demux"
    demuxed_paths = disc_demuxer.demux(
        discs=detected_discs,
        selected_titles=selected_titles,
        demux_dir=demux_dir,
        reporter=reporter,
    )

    dvd_demuxed = _dvd_demuxed_paths(detected_discs, selected_titles, demuxed_paths)
    sar_override_paths: set[Path] = set()
    grain_overrides: dict[Path, bool] = {}

    file_infos = _probe_file_infos(demuxed_paths, ffmpeg_adapter)
    sd_files = _sd_grain_files(file_infos)

    # Show the file selector when a DVD needs a SAR toggle, when there are
    # multiple files to pick from, or when any SD file offers a grain toggle.
    if dvd_demuxed or len(demuxed_paths) > 1 or sd_files:
        file_selection = _run_file_selector(
            file_infos=file_infos,
            dvd_files=dvd_demuxed,
            sd_files=sd_files,
            ffmpeg_adapter=ffmpeg_adapter,
            mpv_adapter=mpv_adapter,
            reporter=reporter,
            file_app_runner=file_app_runner,
        )
        if file_selection is not None:
            demuxed_paths = file_selection.selected
            sar_override_paths = file_selection.sar_override
            grain_overrides = file_selection.grain

    return demux_dir, demuxed_paths, sar_override_paths, grain_overrides


# ---------------------------------------------------------------------------
# Plan wiring helpers
# ---------------------------------------------------------------------------


def _append_demuxed_scan_results(
    scan_results: list[ScanResult],
    demuxed_paths: list[Path],
    output: Path,
) -> None:
    """Append a ScanResult entry for each demuxed MKV so it flows through the pipeline."""
    scan_results.extend(
        ScanResult(
            main_file=mkv_path,
            satellite_files=[],
            output_path=output / mkv_path.stem / (mkv_path.stem + ".mkv"),
        )
        for mkv_path in demuxed_paths
    )


def _apply_demux_dir_to_plan(plan_obj: Plan, demux_dir: Path | None) -> None:
    """Record the demux directory on the Plan if disc demux actually happened."""
    if demux_dir is not None:
        plan_obj.demux_dir = str(demux_dir)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(log_dir: Path, *, console: bool = True) -> None:
    """Create furnace.log in log_dir. Optionally add console output for INFO+."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File: everything (DEBUG+)
    log_path = log_dir / "furnace.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)

    if not console:
        return

    # Console: INFO+ with short format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[furnace] %(message)s"))
    root.addHandler(console_handler)


@app.command()
def plan(
    source: Path = typer.Argument(..., help="Video file or directory"),
    output: Path = typer.Option(..., "-o", help="Output directory"),
    audio_lang: str = typer.Option(
        ..., "--audio-lang", "-al", help="Audio languages, comma-separated (e.g. jpn or rus,eng)"
    ),
    sub_lang: str = typer.Option(..., "--sub-lang", "-sl", help="Subtitle languages, comma-separated (e.g. rus,eng)"),
    names: Path | None = typer.Option(None, "--names", help="Rename map file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without saving"),
    metrics: bool = typer.Option(
        False,
        "--metrics",
        "--vmaf",
        help="Compute perceptual quality metrics per encode: SSIMULACRA2 + Butteraugli "
        "(both paths), CVVDP on the SVT-AV1 grain path, plus VMAF on the NVEnc path. "
        "'--vmaf' is a deprecated alias.",
    ),
    copy_video: bool = typer.Option(
        False, "--copy-video", "-cv", help="Copy eligible video streams verbatim instead of re-encoding"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Process files even if already encoded by Furnace or output exists"
    ),
    jobs: int | None = typer.Option(
        None, "--jobs", "-j", help="Parallel analysis workers (default: CPU cores - 2)"
    ),
    ignore_langs: bool = typer.Option(
        False,
        "--ignore-langs",
        "-il",
        help="Treat all track languages as unspecified (for files with wrong language tags); "
        "reassign per-track in the TUI with 'l'",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Scan source, show TUI for track selection, save JSON plan."""
    audio_lang_list = [x.strip() for x in audio_lang.split(",") if x.strip()]
    sub_lang_list = [x.strip() for x in sub_lang.split(",") if x.strip()]

    cfg = load_config(config)

    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output, console=False)  # console handler removed; reporter owns terminal

    logger.debug(
        "plan command started: source=%s output=%s audio_lang=%s sub_lang=%s names=%s "
        "dry_run=%s metrics=%s copy_video=%s force=%s ignore_langs=%s",
        source,
        output,
        audio_lang,
        sub_lang,
        names,
        dry_run,
        metrics,
        copy_video,
        force,
        ignore_langs,
    )

    reporter = RichPlanReporter(source=source, output=output)
    reporter.start()

    try:
        log_dir = output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe, log_dir=log_dir)
        mpv_adapter = MpvAdapter(cfg.mpv)
        eac3to_adapter = Eac3toAdapter(cfg.eac3to, log_dir=log_dir)
        makemkv_adapter = MakemkvAdapter(cfg.makemkvcon, log_dir=log_dir)

        disc_demuxer = DiscDemuxer(
            bd_port=eac3to_adapter,
            dvd_port=makemkv_adapter,
            mkvmerge_path=cfg.mkvmerge,
            pcm_transcoder=eac3to_adapter,
            prober=ffmpeg_adapter,
        )

        detected_discs = disc_demuxer.detect(source)
        disc_titles: dict[DiscSource, list[DiscTitle]] = {}
        for disc in detected_discs:
            try:
                rel = disc.path.parent.relative_to(source)
                rel_str = str(rel) if str(rel) != "." else disc.path.parent.name
            except ValueError:
                rel_str = disc.path.parent.name
            reporter.detect_disc(disc.disc_type, rel_str)
            titles = disc_demuxer.list_titles(disc)
            reporter.detect_disc_titles_done(len(titles))
            disc_titles[disc] = titles

        demux_dir: Path | None = None
        demuxed_paths: list[Path] = []
        sar_override_paths: set[Path] = set()
        disc_grain_overrides: dict[Path, bool] = {}

        if not dry_run:
            demux_dir, demuxed_paths, sar_override_paths, disc_grain_overrides = _run_disc_demux_interactive(
                source=source,
                detected_discs=detected_discs,
                disc_titles=disc_titles,
                disc_demuxer=disc_demuxer,
                ffmpeg_adapter=ffmpeg_adapter,
                mpv_adapter=mpv_adapter,
                reporter=reporter,
            )

        names_map: dict[str, str] | None = None
        if names is not None:
            with names.open("r", encoding="utf-8") as f:
                names_map = json.load(f)

        scanner = Scanner(prober=ffmpeg_adapter, reporter=reporter)
        scan_results = scanner.scan(source, output, names_map)

        # Plain-files grain flow: with no disc demux, plain SD sources still need
        # the file selector so the user can confirm/override the grain verdict.
        # Probe the scanned main files; if any is SD, show the same selector
        # (no DVDs -> the SAR hint stays hidden). HD-only sources skip it, which
        # preserves the previous "no screen without discs" behaviour.
        plain_grain_overrides: dict[Path, bool] = {}
        if not dry_run and scan_results:
            plain_infos = _probe_file_infos([sr.main_file for sr in scan_results], ffmpeg_adapter)
            plain_sd_files = _sd_grain_files(plain_infos)
            if plain_sd_files:
                plain_selection = _run_file_selector(
                    file_infos=plain_infos,
                    dvd_files=set(),
                    sd_files=plain_sd_files,
                    ffmpeg_adapter=ffmpeg_adapter,
                    mpv_adapter=mpv_adapter,
                    reporter=reporter,
                    file_app_runner=_run_screen_app,
                )
                if plain_selection is not None:
                    plain_grain_overrides = plain_selection.grain
                    selected_plain = set(plain_selection.selected)
                    scan_results = [sr for sr in scan_results if sr.main_file in selected_plain]

        _append_demuxed_scan_results(scan_results, demuxed_paths, output)
        # The appended demuxed entries also deserve scan_file events
        for mkv_path in demuxed_paths:
            reporter.scan_file(mkv_path.name)

        workers = max(1, jobs) if jobs is not None else max(1, (os.cpu_count() or 1) - 2)
        analyzer = Analyzer(prober=ffmpeg_adapter, force=force)
        pipeline = AnalysisPipeline(
            analyzer=analyzer,
            prober=ffmpeg_adapter,
            reporter=reporter,
            max_workers=workers,
        )
        batch = pipeline.run(scan_results, copy_video=copy_video, dry_run=dry_run)
        movies_with_paths = batch.movies
        precomputed_crops = batch.crops

        downmix_overrides: dict[tuple[Path, int], DownmixMode] = {}
        lang_overrides: dict[tuple[Path, int], str] = {}

        def _track_selector(movie: Movie, candidates: list[Track], track_type: TrackType) -> list[Track]:
            lang_list = audio_lang_list if track_type == TrackType.AUDIO else sub_lang_list
            return _select_tracks_tui_for_planner(
                movie,
                candidates,
                track_type,
                mpv_adapter,
                downmix_overrides,
                lang_overrides,
                allow_relabel=ignore_langs,
                lang_list=lang_list,
            )

        def _und_resolver(movie: Movie, track: Track, lang_list: list[str]) -> str:
            return _resolve_und_language_tui(movie, track, lang_list, mpv_adapter)

        if not dry_run:
            reporter.pause()
        planner = PlannerService(
            previewer=mpv_adapter,
            track_selector=_track_selector if not dry_run else None,
            und_resolver=_und_resolver if not dry_run else None,
            reporter=reporter,
            ignore_langs=ignore_langs,
        )
        if not dry_run:
            reporter.resume()

        # Merge the disc and plain-files grain decisions. An empty merge means no
        # interactive screen ran (dry-run or HD-only), so pass None and let the
        # analyzer's per-file verdict rule.
        grain_overrides: dict[Path, bool] | None = None
        if not dry_run:
            grain_overrides = {**disc_grain_overrides, **plain_grain_overrides} or None

        plan_obj = planner.create_plan(
            movies=movies_with_paths,
            audio_lang_filter=audio_lang_list,
            sub_lang_filter=sub_lang_list,
            vmaf_enabled=metrics,
            sar_overrides=sar_override_paths,
            downmix_overrides=downmix_overrides,
            lang_overrides=lang_overrides,
            precomputed_crops=precomputed_crops,
            grain_overrides=grain_overrides,
            copy_video=copy_video,
        )
        _apply_demux_dir_to_plan(plan_obj, demux_dir)

        if dry_run:
            reporter.plan_saved(output / "furnace-plan.json", len(plan_obj.jobs))
        else:
            plan_path = output / "furnace-plan.json"
            save_plan(plan_obj, plan_path)
            reporter.plan_saved(plan_path, len(plan_obj.jobs))

        logger.debug("plan command finished: jobs=%d", len(plan_obj.jobs))
    except KeyboardInterrupt:
        reporter.interrupted()
        raise typer.Exit(code=130) from None
    finally:
        reporter.stop()


@app.command()
def run(
    plan_file: Path = typer.Argument(..., help="JSON plan file"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Read plan and encode all pending jobs."""
    # 1. Load config
    cfg = load_config(config)

    # 2. Load plan (need destination for log dir)
    plan_obj = load_plan(plan_file)

    # 3. Setup file logging -> destination/furnace.log (console OFF — Textual owns terminal)
    destination = Path(plan_obj.destination)
    destination.mkdir(parents=True, exist_ok=True)
    _setup_logging(destination, console=False)

    logger.debug("run command started: plan_file=%s", plan_file)

    pending_count = sum(1 for j in plan_obj.jobs if j.status.value in ("pending", "error"))

    # 4. ESC handling: RunApp binds ESC via Textual; shutdown_event shared with executor
    shutdown_event = threading.Event()
    log_dir = destination / "logs"

    # 5. Define executor factory — RunApp calls this in a worker thread,
    #    passing itself as the progress object.
    def _run_executor(progress: RunApp) -> None:
        tool_output = progress.add_tool_line

        ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe, on_output=tool_output)
        eac3to_adapter = Eac3toAdapter(cfg.eac3to, on_output=tool_output)
        qaac_adapter = QaacAdapter(cfg.qaac64, on_output=tool_output)
        mkvmerge_adapter = MkvmergeAdapter(cfg.mkvmerge, on_output=tool_output)
        mkvpropedit_adapter = MkvpropeditAdapter(cfg.mkvpropedit, on_output=tool_output)
        mkclean_adapter = MkcleanAdapter(cfg.mkclean, on_output=tool_output)
        nvencc_adapter = NVEncCAdapter(cfg.nvencc, on_output=tool_output)
        vship_metrics: VshipMetricsAdapter | None = None
        if cfg.bestsource is not None and cfg.vship is not None:
            vship_metrics = VshipMetricsAdapter(cfg.bestsource, cfg.vship)
        svt_adapter = SvtAv1Adapter(cfg.ffmpeg, on_output=tool_output, metrics=vship_metrics)

        dovi_adapter: DoviToolAdapter | None = None
        if cfg.dovi_tool is not None:
            dovi_adapter = DoviToolAdapter(
                cfg.dovi_tool, cfg.ffmpeg, on_output=tool_output,
            )

        executor = Executor(
            encoder=nvencc_adapter,
            grain_encoder=svt_adapter,
            audio_extractor=ffmpeg_adapter,
            audio_decoder=eac3to_adapter,
            aac_encoder=qaac_adapter,
            muxer=mkvmerge_adapter,
            tagger=mkvpropedit_adapter,
            cleaner=mkclean_adapter,
            prober=ffmpeg_adapter,
            dovi_processor=dovi_adapter,
            video_copier=ffmpeg_adapter,
            progress=progress,
            log_dir=log_dir,
        )
        try:
            executor.run(plan_obj, plan_file)
        finally:
            progress.stop()

    # 6. Run the Textual app (blocks until all jobs done or ESC)
    run_app = RunApp(
        total_jobs=pending_count,
        shutdown_event=shutdown_event,
        executor_fn=_run_executor,
        vmaf_enabled=plan_obj.vmaf_enabled,
    )
    run_app.run()

    # If user requested shutdown (ESC/Ctrl+Q), exit immediately
    # to avoid waiting for worker thread cleanup
    if shutdown_event.is_set():
        os._exit(0)

    # 7. Reload plan from disk (executor updates JSON after each job)
    plan_obj = load_plan(plan_file)

    # 8. ReportPrinter.print_report() — after TUI exits, console is free
    console = Console()
    printer = ReportPrinter()
    printer.print_report(plan_obj, console)

    # Cleanup demux directory after successful run
    if plan_obj.demux_dir:
        demux_path = Path(plan_obj.demux_dir)
        if demux_path.exists():
            all_done = all(j.status == JobStatus.DONE for j in plan_obj.jobs)
            if all_done:
                shutil.rmtree(demux_path, ignore_errors=True)
                logger.info("Cleaned up demux directory: %s", demux_path)

    logger.debug("run command finished")


@app.command()
def scan(
    src: Path = typer.Argument(..., help="Video file or directory to scan"),
    not_encoded: bool = typer.Option(
        False, "--not-encoded", help="Show files with no parseable Furnace tag"
    ),
    encoded: bool = typer.Option(
        False, "--encoded", help="Show files encoded by any Furnace version"
    ),
    max_version: str | None = typer.Option(
        None, "--max-version", help="Show Furnace files at version <= X.Y.Z"
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Inventory video files and their Furnace-encode status (read-only).

    Filter flags select on encode status and union (OR): no flag shows every
    video file. The table goes to stdout (redirect-safe); the summary and any
    warnings go to stderr.
    """
    max_version_tuple: tuple[int, int, int] | None = None
    if max_version is not None:
        try:
            max_version_tuple = parse_version_arg(max_version)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{max_version!r} is not a valid X.Y.Z version", param_hint="--max-version"
            ) from exc

    cfg = load_config(config)
    prober = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe)
    service = ScanService(prober=prober)

    rows, total = service.scan(
        src,
        not_encoded=not_encoded,
        encoded=encoded,
        max_version=max_version_tuple,
    )
    warnings = [f"could not read {row.path}" for row in rows if row.unreadable]
    render_scan_table(rows, root=src, total=total, warnings=warnings)
