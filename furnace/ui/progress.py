"""Post-run report printer (Rich)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from furnace.core.models import Job, JobStatus, Plan
from furnace.ui.fmt import fmt_size

# VMAF (0-100) label thresholds for the post-run summary table.
_VMAF_EXCELLENT = 95
_VMAF_GOOD = 85
_VMAF_FAIR = 70

# SSIMULACRA2 (0-100) label thresholds. Official scale anchors: 90 = visually
# lossless, 80 = very high, 70 = high quality.
_S2_EXCELLENT = 90
_S2_GOOD = 80
_S2_FAIR = 70


def _quality_label(value: float, excellent: float, good: float, fair: float) -> str:
    """Map a higher-is-better score to an excellent/good/fair/poor label."""
    if value >= excellent:
        return "excellent"
    if value >= good:
        return "good"
    if value >= fair:
        return "fair"
    return "poor"


class ReportPrinter:
    """Print a final summary report after all jobs complete."""

    def print_report(self, plan: Plan, console: Console) -> None:
        done_jobs = [j for j in plan.jobs if j.status == JobStatus.DONE]
        error_jobs = [j for j in plan.jobs if j.status == JobStatus.ERROR]
        pending_jobs = [j for j in plan.jobs if j.status == JobStatus.PENDING]

        total = len(plan.jobs)
        n_done = len(done_jobs)
        n_error = len(error_jobs)
        n_skipped = len(pending_jobs)

        total_source = sum(j.source_size for j in done_jobs if j.source_size)
        total_output = sum(j.output_size for j in done_jobs if j.output_size is not None)

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Files processed:", str(n_done))
        summary.add_row("Files skipped:", str(n_skipped))
        summary.add_row("Files with errors:", str(n_error))
        summary.add_row("Total files:", str(total))
        console.print(summary)
        console.print()

        if done_jobs and total_source > 0:
            size_table = Table.grid(padding=(0, 2))
            size_table.add_column(style="bold")
            size_table.add_column()
            size_table.add_row("Total source size:", fmt_size(total_source))
            if total_output:
                size_table.add_row("Total output size:", fmt_size(total_output))
                saved = total_source - total_output
                pct = saved / total_source * 100
                sign = "-" if saved >= 0 else "+"
                savings_str = f"{fmt_size(abs(saved))} ({sign}{abs(pct):.1f}%)"
                size_table.add_row("Space saved:", savings_str)
            console.print(size_table)
            console.print()

        # Per-file results
        if done_jobs:
            console.print("[bold]Files:[/bold]")
            for job in done_jobs:
                name = Path(job.source_files[0]).name if job.source_files else Path(job.output_file).name
                src = fmt_size(job.source_size)
                out = fmt_size(job.output_size)
                quality_str = ""
                if job.vmaf_score is not None:
                    label = _quality_label(job.vmaf_score, _VMAF_EXCELLENT, _VMAF_GOOD, _VMAF_FAIR)
                    quality_str += f"  VMAF {job.vmaf_score:.1f} ({label})"
                if job.ssimulacra2_score is not None:
                    label = _quality_label(job.ssimulacra2_score, _S2_EXCELLENT, _S2_GOOD, _S2_FAIR)
                    quality_str += f"  SSIMU2 {job.ssimulacra2_score:.2f} ({label})"
                if job.butteraugli_score is not None:
                    quality_str += f"  Butteraugli {job.butteraugli_score:.2f}"
                if job.cvvdp_score is not None:
                    quality_str += f"  CVVDP {job.cvvdp_score:.2f}"
                console.print(f"  {name}  {src} -> {out}{quality_str}")
            console.print()

        if plan.vmaf_enabled:
            self._print_metric_averages(done_jobs, console)

        if error_jobs:
            console.print("[bold red]Errors:[/bold red]")
            for job in error_jobs:
                name = Path(job.output_file).name
                err = job.error or "unknown error"
                console.print(f"  [red]{name}[/red]: {err}")
            console.print()

    def _print_metric_averages(self, done_jobs: list[Job], console: Console) -> None:
        """Print the average of each quality metric any completed job recorded."""
        specs: list[tuple[str, list[float | None]]] = [
            ("VMAF", [j.vmaf_score for j in done_jobs]),
            ("SSIMU2", [j.ssimulacra2_score for j in done_jobs]),
            ("Butteraugli", [j.butteraugli_score for j in done_jobs]),
            ("CVVDP", [j.cvvdp_score for j in done_jobs]),
        ]
        parts: list[str] = []
        for label, raw in specs:
            scores = [s for s in raw if s is not None]
            if scores:
                avg = sum(scores) / len(scores)
                parts.append(f"{label} {avg:.2f} (n={len(scores)})")
        if parts:
            console.print("[bold]Average:[/bold] " + "  |  ".join(parts))
            console.print()
