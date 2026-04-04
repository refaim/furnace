from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import typer

app = typer.Typer()

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: Path, *, console: bool = True) -> None:
    """Create furnace.log in log_dir. Optionally add console output for INFO+."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File: everything (DEBUG+)
    log_path = log_dir / "furnace.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(file_handler)

    if not console:
        return

    # Console: INFO+ with short format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[furnace] %(message)s"))
    root.addHandler(console_handler)


def _start_esc_watcher(shutdown_event: threading.Event) -> None:
    """Watch for ESC key press and set shutdown_event when detected.

    Uses msvcrt on Windows. On non-Windows platforms this function is a no-op.
    """
    import sys
    if sys.platform != "win32":
        return

    import msvcrt

    def _watch() -> None:
        while not shutdown_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\x1b":  # ESC
                    shutdown_event.set()
                    return
            time.sleep(0.05)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


@app.command()
def plan(
    source: Path = typer.Argument(..., help="Video file or directory"),
    output: Path = typer.Option(..., "-o", help="Output directory"),
    audio_lang: str = typer.Option(..., "--audio-lang", "-al", help="Audio languages, comma-separated (e.g. jpn or rus,eng)"),
    sub_lang: str = typer.Option(..., "--sub-lang", "-sl", help="Subtitle languages, comma-separated (e.g. rus,eng)"),
    names: Path | None = typer.Option(None, "--names", help="Rename map file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without saving"),
    vmaf: bool = typer.Option(False, "--vmaf", help="Enable VMAF"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Scan source, show TUI for track selection, save JSON plan."""
    # Parse comma-separated language lists
    audio_lang_list = [x.strip() for x in audio_lang.split(",") if x.strip()]
    sub_lang_list = [x.strip() for x in sub_lang.split(",") if x.strip()]

    # Lazy imports to avoid circular imports
    from .adapters.eac3to import Eac3toAdapter
    from .adapters.ffmpeg import FFmpegAdapter
    from .adapters.makemkv import MakemkvAdapter
    from .adapters.mpv import MpvAdapter
    from .config import load_config
    from .core.models import DiscTitle, DiscSource, DiscType, Movie, ScanResult, Track, TrackType
    from .plan import save_plan
    from .services.analyzer import Analyzer
    from .services.disc_demuxer import DiscDemuxer
    from .services.planner import PlannerService
    from .services.scanner import Scanner
    from .ui.tui import FileSelection, FileSelectorScreen, PlaylistSelectorScreen, TrackSelectorScreen

    # 1. Load config
    cfg = load_config(config)

    # 2. Setup file logging -> output/furnace.log
    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output)

    logger.debug(
        "plan command started: source=%s output=%s audio_lang=%s sub_lang=%s names=%s dry_run=%s vmaf=%s",
        source, output, audio_lang, sub_lang, names, dry_run, vmaf,
    )

    # 3. Create adapters (only those needed for planning)
    ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe)
    mpv_adapter = MpvAdapter(cfg.mpv)

    import re
    import sys
    _progress_re = re.compile(r"(?:Current progress|^\d+%$|^analyze:|^process:|^Progress:)", re.IGNORECASE)
    _pct_re = re.compile(r"(\d+)%")

    def _console_output(line: str) -> None:
        """Print tool output to console. Progress lines shown inline on one line."""
        if _progress_re.search(line):
            pct_match = _pct_re.search(line)
            if pct_match:
                sys.stderr.write(f"\r[furnace] {pct_match.group(1)}%  ")
                sys.stderr.flush()
        else:
            sys.stderr.write("\r\033[K")  # clear progress line
            sys.stderr.flush()
            typer.echo(line)

    eac3to_adapter = Eac3toAdapter(cfg.eac3to, on_output=_console_output)
    makemkv_adapter = MakemkvAdapter(cfg.makemkvcon, on_output=_console_output)

    # 4. Disc demux phase
    disc_demuxer = DiscDemuxer(bd_port=eac3to_adapter, dvd_port=makemkv_adapter, mkvmerge_path=cfg.mkvmerge)
    typer.echo("[furnace] Scanning for disc structures...")
    detected_discs = disc_demuxer.detect(source)
    demux_dir: Path | None = None
    demuxed_paths: list[Path] = []
    sar_override_paths: set[Path] = set()

    if detected_discs and not dry_run:
        from textual.app import App, ComposeResult
        from textual.widgets import Header

        typer.echo(f"[furnace] Found {len(detected_discs)} disc(s)")

        # For each disc, list titles and let user pick
        selected_titles: dict[DiscSource, list[DiscTitle]] = {}
        for disc in detected_discs:
            typer.echo(f"[furnace] Listing titles for {disc.disc_type.value.upper()}: {disc.path}")
            playlists = disc_demuxer.list_titles(disc)
            if not playlists:
                logger.warning("No playlists found for disc at %s", disc.path)
                continue

            # Auto-select if only one title, otherwise show TUI
            if len(playlists) == 1:
                selected_titles[disc] = playlists
                continue

            disc_label = disc.path.parent.name
            selected_playlists_for_disc: list[DiscTitle] = []

            def _run_playlist_selector(
                _disc_label: str = disc_label,
                _playlists: list[DiscTitle] = playlists,
            ) -> None:
                nonlocal selected_playlists_for_disc

                class _PlaylistApp(App[list[DiscTitle]]):
                    def compose(self) -> ComposeResult:
                        yield Header()

                    def on_mount(self) -> None:
                        def _on_dismiss(result: list[DiscTitle] | None) -> None:
                            nonlocal selected_playlists_for_disc
                            selected_playlists_for_disc = result or []
                            self.exit(selected_playlists_for_disc)

                        self.push_screen(
                            PlaylistSelectorScreen(disc_label=_disc_label, playlists=_playlists),
                            _on_dismiss,
                        )

                _PlaylistApp().run()

            _run_playlist_selector()
            if selected_playlists_for_disc:
                selected_titles[disc] = selected_playlists_for_disc

        # Demux selected titles
        if selected_titles:
            total_titles = sum(len(t) for t in selected_titles.values())
            typer.echo(f"[furnace] Demuxing {total_titles} title(s)...")
            demux_dir = source / ".furnace_demux"
            demuxed_paths = disc_demuxer.demux(
                discs=detected_discs,
                selected_titles=selected_titles,
                demux_dir=demux_dir,
                on_progress=_console_output,
            )

            # Track which demuxed files came from DVD
            dvd_demuxed: set[Path] = set()
            for disc in detected_discs:
                if disc.disc_type == DiscType.DVD and disc in selected_titles:
                    for p in demuxed_paths:
                        disc_label = disc.path.parent.name
                        if p.name.startswith(disc_label):
                            dvd_demuxed.add(p)

            # Show file selector for DVD files (SAR override) or if >1 file
            if dvd_demuxed or len(demuxed_paths) > 1:
                file_infos: list[tuple[Path, float, int]] = []
                for mkv_path in demuxed_paths:
                    probe_data = ffmpeg_adapter.probe(mkv_path)
                    fmt = probe_data.get("format", {})
                    duration_s = float(fmt.get("duration", 0))
                    size_bytes = int(fmt.get("size", 0))
                    file_infos.append((mkv_path, duration_s, size_bytes))

                file_selection: FileSelection | None = None

                class _FileApp(App[FileSelection]):
                    def compose(self) -> ComposeResult:
                        yield Header()

                    def on_mount(self) -> None:
                        def _on_dismiss(result: FileSelection | None) -> None:
                            nonlocal file_selection
                            file_selection = result
                            self.exit(result)

                        self.push_screen(
                            FileSelectorScreen(
                                files=file_infos,
                                dvd_files=dvd_demuxed,
                                preview_cb=lambda p, a: mpv_adapter.preview_file(p, aspect_override=a),
                            ),
                            _on_dismiss,
                        )

                _FileApp().run()
                if file_selection is not None:
                    demuxed_paths = file_selection.selected
                    sar_override_paths = file_selection.sar_override

    # 5. Load names map if provided
    names_map: dict[str, str] | None = None
    if names is not None:
        import json
        with names.open("r", encoding="utf-8") as f:
            names_map = json.load(f)

    # 6. Scanner.scan() -> list[ScanResult]
    scanner = Scanner(prober=ffmpeg_adapter)
    scan_results = scanner.scan(source, output, names_map)

    # Add demuxed MKVs as extra ScanResult entries
    for mkv_path in demuxed_paths:
        scan_results.append(ScanResult(
            main_file=mkv_path,
            satellite_files=[],
            output_path=output / mkv_path.stem / (mkv_path.stem + ".mkv"),
        ))

    # 7. Analyzer.analyze() -> list[tuple[Movie, Path]]
    analyzer = Analyzer(prober=ffmpeg_adapter)
    movies_with_paths: list[tuple[Movie, Path]] = []
    for sr in scan_results:
        movie = analyzer.analyze(sr)
        if movie is not None:
            # Apply SAR override for DVD files marked by user
            if sr.main_file in sar_override_paths:
                movie.video.sar_num = 64
                movie.video.sar_den = 45
            movies_with_paths.append((movie, sr.output_path))

    # 8. PlannerService.create_plan() with TUI track selector
    def _select_tracks_tui(movie: Movie, candidates: list[Track], track_type: TrackType) -> list[Track]:
        """Run Textual TrackSelectorScreen synchronously for user to pick tracks."""
        from textual.app import App, ComposeResult
        from textual.widgets import Header

        selected: list[Track] = []

        class _SelectorApp(App[list[Track]]):
            def compose(self) -> ComposeResult:
                yield Header()

            def on_mount(self) -> None:
                def _on_dismiss(result: list[Track] | None) -> None:
                    nonlocal selected
                    selected = result or []
                    self.exit(selected)

                self.push_screen(
                    TrackSelectorScreen(
                        movie=movie,
                        tracks=candidates,
                        track_type=track_type,
                        preview_cb=None,
                    ),
                    _on_dismiss,
                )

        _SelectorApp().run()
        return selected

    planner = PlannerService(
        prober=ffmpeg_adapter,
        previewer=mpv_adapter,
        track_selector=_select_tracks_tui if not dry_run else None,
    )
    plan_obj = planner.create_plan(
        movies=movies_with_paths,
        audio_lang_filter=audio_lang_list,
        sub_lang_filter=sub_lang_list,
        vmaf_enabled=vmaf,
        dry_run=dry_run,
    )

    # Set demux_dir on the plan if disc demux happened
    if demux_dir is not None:
        plan_obj.demux_dir = str(demux_dir)

    # 9. save_plan() if not dry_run
    if dry_run:
        pending = [j for j in plan_obj.jobs if j.status.value == "pending"]
        typer.echo(f"Dry run: {len(plan_obj.jobs)} job(s) planned, {len(pending)} pending. Plan NOT saved.")
    else:
        plan_path = output / "furnace-plan.json"
        save_plan(plan_obj, plan_path)
        pending = [j for j in plan_obj.jobs if j.status.value == "pending"]
        typer.echo(f"Plan saved: {plan_path}")
        typer.echo(f"Jobs: {len(plan_obj.jobs)} total, {len(pending)} pending")

    logger.debug("plan command finished: jobs=%d", len(plan_obj.jobs))


@app.command()
def run(
    plan_file: Path = typer.Argument(..., help="JSON plan file"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Read plan and encode all pending jobs."""
    # Lazy imports to avoid circular imports
    from .adapters.eac3to import Eac3toAdapter
    from .adapters.ffmpeg import FFmpegAdapter
    from .adapters.mkclean import MkcleanAdapter
    from .adapters.mkvmerge import MkvmergeAdapter
    from .adapters.mkvpropedit import MkvpropeditAdapter
    from .adapters.qaac import QaacAdapter
    from .config import load_config
    from .plan import load_plan
    from .services.executor import Executor
    from .ui.progress import ReportPrinter
    from .ui.run_tui import RunApp

    # 1. Load config
    cfg = load_config(config)

    # 2. Load plan (need destination for log dir)
    plan_obj = load_plan(plan_file)

    # 3. Setup file logging -> destination/furnace.log (console OFF — Textual owns terminal)
    destination = Path(plan_obj.destination)
    destination.mkdir(parents=True, exist_ok=True)
    _setup_logging(destination, console=False)

    logger.debug("run command started: plan_file=%s", plan_file)

    pending_count = sum(
        1 for j in plan_obj.jobs
        if j.status.value in ("pending", "error")
    )

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

        executor = Executor(
            encoder=ffmpeg_adapter,
            audio_extractor=ffmpeg_adapter,
            audio_decoder=eac3to_adapter,
            aac_encoder=qaac_adapter,
            muxer=mkvmerge_adapter,
            tagger=mkvpropedit_adapter,
            cleaner=mkclean_adapter,
            prober=ffmpeg_adapter,
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

    # 7. Reload plan from disk (executor updates JSON after each job)
    plan_obj = load_plan(plan_file)

    # 8. ReportPrinter.print_report() — after TUI exits, console is free
    from rich.console import Console
    console = Console()
    printer = ReportPrinter()
    printer.print_report(plan_obj, console)

    # Cleanup demux directory after successful run
    if plan_obj.demux_dir:
        from .core.models import JobStatus
        demux_path = Path(plan_obj.demux_dir)
        if demux_path.exists():
            all_done = all(j.status == JobStatus.DONE for j in plan_obj.jobs)
            if all_done:
                import shutil
                shutil.rmtree(demux_path, ignore_errors=True)
                logger.info("Cleaned up demux directory: %s", demux_path)

    logger.debug("run command finished")
