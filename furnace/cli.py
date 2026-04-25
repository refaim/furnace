from __future__ import annotations

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
from .config import load_config
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
from .plan import load_plan, save_plan
from .services.analyzer import Analyzer
from .services.disc_demuxer import DiscDemuxer
from .services.executor import Executor
from .services.planner import PlannerService
from .services.scanner import Scanner
from .ui.plan_console import RichPlanReporter
from .ui.progress import ReportPrinter
from .ui.run_tui import RunApp
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
    app_runner: Callable[[Callable[[], Screen[TrackSelection]]], TrackSelection | None] = _run_screen_app,
) -> TrackSelection:
    """Run Textual TrackSelectorScreen synchronously for user to pick tracks."""

    def _factory() -> Screen[TrackSelection]:
        return TrackSelectorScreen(
            movie=movie,
            tracks=candidates,
            track_type=track_type,
            preview_cb=_make_preview_track_cb(movie, mpv_adapter),
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
    *,
    app_runner: Callable[[Callable[[], Screen[TrackSelection]]], TrackSelection | None] = _run_screen_app,
) -> list[Track]:
    """Planner-facing wrapper: returns list[Track] and mutates downmix_overrides for audio."""
    result = _select_tracks_tui(movie, candidates, track_type, mpv_adapter, app_runner=app_runner)
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


def _probe_file_infos(demuxed_paths: list[Path], ffmpeg_adapter: FFmpegAdapter) -> list[tuple[Path, float, int]]:
    """Probe each demuxed file for duration/size for the file-selector UI."""
    file_infos: list[tuple[Path, float, int]] = []
    for mkv_path in demuxed_paths:
        probe_data = ffmpeg_adapter.probe(mkv_path)
        fmt = probe_data.get("format", {})
        duration_s = float(fmt.get("duration", 0))
        size_bytes = int(fmt.get("size", 0))
        file_infos.append((mkv_path, duration_s, size_bytes))
    return file_infos


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
) -> tuple[Path | None, list[Path], set[Path]]:
    """Coordinate the interactive disc demux flow.

    Returns `(demux_dir, demuxed_paths, sar_override_paths)`. When no discs are
    provided, returns `(None, [], set())`.
    """
    if not detected_discs:
        return None, [], set()

    selected_titles = _collect_selected_titles(
        detected_discs,
        disc_titles,
        reporter=reporter,
        playlist_app_runner=playlist_app_runner,
    )

    if not selected_titles:
        return None, [], set()

    demux_dir = source / ".furnace_demux"
    demuxed_paths = disc_demuxer.demux(
        discs=detected_discs,
        selected_titles=selected_titles,
        demux_dir=demux_dir,
        reporter=reporter,
    )

    dvd_demuxed = _dvd_demuxed_paths(detected_discs, selected_titles, demuxed_paths)
    sar_override_paths: set[Path] = set()

    if dvd_demuxed or len(demuxed_paths) > 1:
        if reporter is not None:
            reporter.pause()
        file_infos = _probe_file_infos(demuxed_paths, ffmpeg_adapter)

        def _factory(
            _file_infos: list[tuple[Path, float, int]] = file_infos,
            _dvd: set[Path] = dvd_demuxed,
        ) -> Screen[FileSelection]:
            return FileSelectorScreen(
                files=_file_infos,
                dvd_files=_dvd,
                preview_cb=lambda p, a: mpv_adapter.preview_file(p, aspect_override=a),
            )

        file_selection = file_app_runner(_factory)
        if reporter is not None:
            reporter.resume()
        if file_selection is not None:
            demuxed_paths = file_selection.selected
            sar_override_paths = file_selection.sar_override

    return demux_dir, demuxed_paths, sar_override_paths


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
    vmaf: bool = typer.Option(False, "--vmaf", help="Enable VMAF"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Scan source, show TUI for track selection, save JSON plan."""
    audio_lang_list = [x.strip() for x in audio_lang.split(",") if x.strip()]
    sub_lang_list = [x.strip() for x in sub_lang.split(",") if x.strip()]

    cfg = load_config(config)

    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output, console=False)  # console handler removed; reporter owns terminal

    logger.debug(
        "plan command started: source=%s output=%s audio_lang=%s sub_lang=%s names=%s dry_run=%s vmaf=%s",
        source,
        output,
        audio_lang,
        sub_lang,
        names,
        dry_run,
        vmaf,
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

        if not dry_run:
            demux_dir, demuxed_paths, sar_override_paths = _run_disc_demux_interactive(
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
        _append_demuxed_scan_results(scan_results, demuxed_paths, output)
        # The appended demuxed entries also deserve scan_file events
        for mkv_path in demuxed_paths:
            reporter.scan_file(mkv_path.name)

        analyzer = Analyzer(prober=ffmpeg_adapter, reporter=reporter)
        movies_with_paths: list[tuple[Movie, Path]] = []
        for sr in scan_results:
            try:
                movie = analyzer.analyze(sr)
            except ValueError as exc:
                # analyze() raises for HDR10+; reporter already saw analyze_file_failed
                logger.warning("analyze raised: %s", exc)
                continue
            if movie is not None:
                movies_with_paths.append((movie, sr.output_path))

        downmix_overrides: dict[tuple[Path, int], DownmixMode] = {}

        def _track_selector(movie: Movie, candidates: list[Track], track_type: TrackType) -> list[Track]:
            return _select_tracks_tui_for_planner(movie, candidates, track_type, mpv_adapter, downmix_overrides)

        def _und_resolver(movie: Movie, track: Track, lang_list: list[str]) -> str:
            return _resolve_und_language_tui(movie, track, lang_list, mpv_adapter)

        if not dry_run:
            reporter.pause()
        planner = PlannerService(
            prober=ffmpeg_adapter,
            previewer=mpv_adapter,
            track_selector=_track_selector if not dry_run else None,
            und_resolver=_und_resolver if not dry_run else None,
            reporter=reporter,
        )
        if not dry_run:
            reporter.resume()

        plan_obj = planner.create_plan(
            movies=movies_with_paths,
            audio_lang_filter=audio_lang_list,
            sub_lang_filter=sub_lang_list,
            vmaf_enabled=vmaf,
            dry_run=dry_run,
            sar_overrides=sar_override_paths,
            downmix_overrides=downmix_overrides,
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

        dovi_adapter: DoviToolAdapter | None = None
        if cfg.dovi_tool is not None:
            dovi_adapter = DoviToolAdapter(cfg.dovi_tool, on_output=tool_output)

        executor = Executor(
            encoder=nvencc_adapter,
            audio_extractor=ffmpeg_adapter,
            audio_decoder=eac3to_adapter,
            aac_encoder=qaac_adapter,
            muxer=mkvmerge_adapter,
            tagger=mkvpropedit_adapter,
            cleaner=mkclean_adapter,
            prober=ffmpeg_adapter,
            dovi_processor=dovi_adapter,
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
