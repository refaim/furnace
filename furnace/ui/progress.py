"""Post-run report printer (Rich)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from furnace.core.models import JobStatus, Plan
from furnace.ui.fmt import fmt_size


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
                console.print(f"  {name}  {src} -> {out}")
            console.print()

        if error_jobs:
            console.print("[bold red]Errors:[/bold red]")
            for job in error_jobs:
                name = Path(job.output_file).name
                err = job.error or "unknown error"
                console.print(f"  [red]{name}[/red]: {err}")
            console.print()
