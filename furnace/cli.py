from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import typer

app = typer.Typer()

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: Path) -> None:
    """Create furnace.log in log_dir with a standard format."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "furnace.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)


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
    lang: list[str] | None = typer.Option(None, "--lang", help="Language filter (e.g. rus eng)"),
    names: Path | None = typer.Option(None, "--names", help="Rename map file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without saving"),
    vmaf: bool = typer.Option(False, "--vmaf", help="Enable VMAF"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Scan source, show TUI for track selection, save JSON plan."""
    # Lazy imports to avoid circular imports
    from .adapters.ffmpeg import FFmpegAdapter
    from .adapters.mpv import MpvAdapter
    from .config import load_config
    from .plan import save_plan
    from .services.analyzer import Analyzer
    from .services.planner import PlannerService
    from .services.scanner import Scanner

    # 1. Load config
    cfg = load_config(config)

    # 2. Setup file logging -> output/furnace.log
    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output)

    logger.info(
        "plan command started: source=%s output=%s lang=%s names=%s dry_run=%s vmaf=%s",
        source, output, lang, names, dry_run, vmaf,
    )

    # 3. Create adapters (only those needed for planning)
    ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe)
    mpv_adapter = MpvAdapter(cfg.mpv)

    # 4. Load names map if provided
    names_map: dict[str, str] | None = None
    if names is not None:
        import json
        with names.open("r", encoding="utf-8") as f:
            names_map = json.load(f)

    # 5. Scanner.scan() -> list[ScanResult]
    scanner = Scanner(prober=ffmpeg_adapter)
    scan_results = scanner.scan(source, output, names_map)

    # 6. Analyzer.analyze() -> list[tuple[Movie, Path]]
    analyzer = Analyzer(prober=ffmpeg_adapter)
    from .core.models import Movie
    movies_with_paths: list[tuple[Movie, Path]] = []
    for sr in scan_results:
        movie = analyzer.analyze(sr)
        if movie is not None:
            movies_with_paths.append((movie, sr.output_path))

    # 7. PlannerService.create_plan() with TUI track selector
    from .core.models import Track, TrackType
    from .ui.tui import TrackSelectorScreen

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
        lang_filter=lang,
        vmaf_enabled=vmaf,
        dry_run=dry_run,
    )

    # 8. save_plan() if not dry_run
    if dry_run:
        pending = [j for j in plan_obj.jobs if j.status.value == "pending"]
        typer.echo(f"Dry run: {len(plan_obj.jobs)} job(s) planned, {len(pending)} pending. Plan NOT saved.")
    else:
        plan_path = output / "furnace-plan.json"
        save_plan(plan_obj, plan_path)
        pending = [j for j in plan_obj.jobs if j.status.value == "pending"]
        typer.echo(f"Plan saved: {plan_path}")
        typer.echo(f"Jobs: {len(plan_obj.jobs)} total, {len(pending)} pending")

    logger.info("plan command finished: jobs=%d", len(plan_obj.jobs))


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

    # 1. Load config
    cfg = load_config(config)

    # 2. Load plan (need destination for log dir)
    plan_obj = load_plan(plan_file)

    # 3. Setup file logging -> destination/furnace.log
    destination = Path(plan_obj.destination)
    destination.mkdir(parents=True, exist_ok=True)
    _setup_logging(destination)

    logger.info("run command started: plan_file=%s", plan_file)

    # 4. Create adapters (all needed for execution)
    ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe)
    eac3to_adapter = Eac3toAdapter(cfg.eac3to)
    qaac_adapter = QaacAdapter(cfg.qaac64)
    mkvmerge_adapter = MkvmergeAdapter(cfg.mkvmerge)
    mkvpropedit_adapter = MkvpropeditAdapter(cfg.mkvpropedit)
    mkclean_adapter = MkcleanAdapter(cfg.mkclean)

    # 5. Start ESC watcher thread
    shutdown_event = threading.Event()
    _start_esc_watcher(shutdown_event)

    # 6. Create console for output
    from rich.console import Console
    console = Console()

    # 7. Executor.run()
    executor = Executor(
        encoder=ffmpeg_adapter,
        audio_extractor=ffmpeg_adapter,
        audio_decoder=eac3to_adapter,
        aac_encoder=qaac_adapter,
        muxer=mkvmerge_adapter,
        tagger=mkvpropedit_adapter,
        cleaner=mkclean_adapter,
        prober=ffmpeg_adapter,
    )
    executor.run(plan_obj, plan_file)

    # 8. ReportPrinter.print_report()
    printer = ReportPrinter()
    printer.print_report(plan_obj, console)

    logger.info("run command finished")
