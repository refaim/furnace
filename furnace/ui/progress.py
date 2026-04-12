"""Post-run report printer (Rich)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from furnace.core.models import JobStatus, Plan

# VMAF (0-100) label thresholds for the post-run summary table.
_VMAF_EXCELLENT = 95
_VMAF_GOOD = 85
_VMAF_FAIR = 70

# SSIM (0-1) label thresholds for the post-run summary table.
_SSIM_EXCELLENT = 0.99
_SSIM_GOOD = 0.95
_SSIM_FAIR = 0.90


def _fmt_size(n: int | None) -> str:
    if n is None or n == 0:
        return "?"
    mb = n / (1024 * 1024)
    return f"{mb:,.0f} MB"


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
            size_table.add_row("Total source size:", _fmt_size(total_source))
            if total_output:
                size_table.add_row("Total output size:", _fmt_size(total_output))
                saved = total_source - total_output
                pct = saved / total_source * 100
                sign = "-" if saved >= 0 else "+"
                savings_str = f"{_fmt_size(abs(saved))} ({sign}{abs(pct):.1f}%)"
                size_table.add_row("Space saved:", savings_str)
            console.print(size_table)
            console.print()

        # Per-file results
        if done_jobs:
            console.print("[bold]Files:[/bold]")
            for job in done_jobs:
                name = Path(job.source_files[0]).name if job.source_files else Path(job.output_file).name
                src = _fmt_size(job.source_size)
                out = _fmt_size(job.output_size)
                quality_str = ""
                if job.vmaf_score is not None:
                    v = job.vmaf_score
                    if v >= _VMAF_EXCELLENT:
                        label = "excellent"
                    elif v >= _VMAF_GOOD:
                        label = "good"
                    elif v >= _VMAF_FAIR:
                        label = "fair"
                    else:
                        label = "poor"
                    quality_str = f"  VMAF {v:.1f} ({label})"
                if job.ssim_score is not None:
                    s = job.ssim_score
                    if s >= _SSIM_EXCELLENT:
                        slabel = "excellent"
                    elif s >= _SSIM_GOOD:
                        slabel = "good"
                    elif s >= _SSIM_FAIR:
                        slabel = "fair"
                    else:
                        slabel = "poor"
                    quality_str += f"  SSIM {s:.4f} ({slabel})"
                console.print(f"  {name}  {src} -> {out}{quality_str}")
            console.print()

        if plan.vmaf_enabled:
            vmaf_scores = [j.vmaf_score for j in done_jobs if j.vmaf_score is not None]
            ssim_scores = [j.ssim_score for j in done_jobs if j.ssim_score is not None]
            if vmaf_scores:
                avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                avg_line = f"[bold]Average VMAF:[/bold] {avg_vmaf:.2f}  (n={len(vmaf_scores)})"
                if ssim_scores:
                    avg_ssim = sum(ssim_scores) / len(ssim_scores)
                    avg_line += f"  |  [bold]SSIM:[/bold] {avg_ssim:.4f}"
                console.print(avg_line)
                console.print()

        if error_jobs:
            console.print("[bold red]Errors:[/bold red]")
            for job in error_jobs:
                name = Path(job.output_file).name
                err = job.error or "unknown error"
                console.print(f"  [red]{name}[/red]: {err}")
            console.print()
