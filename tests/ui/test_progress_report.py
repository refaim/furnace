from __future__ import annotations

from io import StringIO

from rich.console import Console

from furnace.core.models import JobStatus
from furnace.ui.progress import ReportPrinter
from tests.conftest import make_job, make_plan


def _render(**plan_kw: object) -> str:
    plan = make_plan(**plan_kw)  # type: ignore[arg-type]
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    ReportPrinter().print_report(plan, console)
    return buf.getvalue()


class TestAllDone:
    def test_shows_processed_count(self) -> None:
        jobs = [
            make_job(job_id="j1", status=JobStatus.DONE, source_size=2_000_000, output_size=1_500_000),
            make_job(job_id="j2", status=JobStatus.DONE, source_size=3_000_000, output_size=2_000_000),
        ]
        text = _render(jobs=jobs)
        assert "Files processed:" in text
        assert "2" in text.split("Files processed:")[1].split("\n")[0]

    def test_shows_sizes(self) -> None:
        jobs = [
            make_job(job_id="j1", status=JobStatus.DONE, source_size=10_485_760, output_size=5_242_880),
        ]
        text = _render(jobs=jobs)
        assert "10 MB" in text
        assert "5 MB" in text

    def test_shows_zero_skipped_and_errors(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000)]
        text = _render(jobs=jobs)
        assert "Files skipped:" in text
        assert "0" in text.split("Files skipped:")[1].split("\n")[0]
        assert "Files with errors:" in text
        assert "0" in text.split("Files with errors:")[1].split("\n")[0]


class TestErrorJobs:
    def test_shows_error_messages(self) -> None:
        jobs = [
            make_job(
                job_id="e1",
                status=JobStatus.ERROR,
                error="encoder crashed",
                output_file="/out/bad.mkv",
            ),
        ]
        text = _render(jobs=jobs)
        assert "Errors:" in text
        assert "bad.mkv" in text
        assert "encoder crashed" in text

    def test_unknown_error_when_no_message(self) -> None:
        jobs = [
            make_job(
                job_id="e2",
                status=JobStatus.ERROR,
                error=None,
                output_file="/out/oops.mkv",
            ),
        ]
        text = _render(jobs=jobs)
        assert "unknown error" in text

    def test_error_count(self) -> None:
        jobs = [
            make_job(job_id="e1", status=JobStatus.ERROR, error="fail1"),
            make_job(job_id="e2", status=JobStatus.ERROR, error="fail2"),
        ]
        text = _render(jobs=jobs)
        assert "2" in text.split("Files with errors:")[1].split("\n")[0]


class TestMixedStatuses:
    def test_done_error_pending_counts(self) -> None:
        jobs = [
            make_job(job_id="d1", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000),
            make_job(job_id="e1", status=JobStatus.ERROR, error="boom"),
            make_job(job_id="p1", status=JobStatus.PENDING),
        ]
        text = _render(jobs=jobs)
        assert "1" in text.split("Files processed:")[1].split("\n")[0]
        assert "1" in text.split("Files skipped:")[1].split("\n")[0]
        assert "1" in text.split("Files with errors:")[1].split("\n")[0]
        assert "3" in text.split("Total files:")[1].split("\n")[0]

    def test_error_section_and_files_section(self) -> None:
        jobs = [
            make_job(
                job_id="d1",
                status=JobStatus.DONE,
                source_files=["/src/good.mkv"],
                source_size=2_000_000,
                output_size=1_000_000,
            ),
            make_job(
                job_id="e1",
                status=JobStatus.ERROR,
                error="encoding failed",
                output_file="/out/bad.mkv",
            ),
        ]
        text = _render(jobs=jobs)
        assert "Files:" in text
        assert "good.mkv" in text
        assert "Errors:" in text
        assert "bad.mkv" in text
        assert "encoding failed" in text


class TestSizeSavings:
    def test_positive_savings(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=10_485_760, output_size=5_242_880)]
        text = _render(jobs=jobs)
        assert "Space saved:" in text
        assert "5 MB" in text
        assert "-50.0%" in text

    def test_no_savings_output_equals_source(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=10_485_760, output_size=10_485_760)]
        text = _render(jobs=jobs)
        assert "Space saved:" in text
        assert "-0.0%" in text

    def test_output_larger_than_source(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=5_242_880, output_size=10_485_760)]
        text = _render(jobs=jobs)
        assert "Space saved:" in text
        assert "+100.0%" in text

    def test_no_size_table_when_source_size_zero(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=0, output_size=500_000)]
        text = _render(jobs=jobs)
        assert "Total source size:" not in text

    def test_no_output_size_row_when_output_is_none(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=10_485_760, output_size=None)]
        text = _render(jobs=jobs)
        assert "Total source size:" in text
        assert "Total output size:" not in text
        assert "Space saved:" not in text

    def test_no_output_size_row_when_output_is_zero(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=10_485_760, output_size=0)]
        text = _render(jobs=jobs)
        assert "Total source size:" in text
        assert "Total output size:" not in text
        assert "Space saved:" not in text


class TestNoDoneJobs:
    def test_all_pending(self) -> None:
        jobs = [
            make_job(job_id="p1", status=JobStatus.PENDING),
            make_job(job_id="p2", status=JobStatus.PENDING),
        ]
        text = _render(jobs=jobs)
        assert "Files processed:" in text
        assert "0" in text.split("Files processed:")[1].split("\n")[0]
        assert "Files:" not in text
        assert "Total source size:" not in text

    def test_all_errors(self) -> None:
        jobs = [
            make_job(job_id="e1", status=JobStatus.ERROR, error="fail"),
        ]
        text = _render(jobs=jobs)
        assert "0" in text.split("Files processed:")[1].split("\n")[0]
        assert "Files:" not in text
        assert "Total source size:" not in text

    def test_empty_plan(self) -> None:
        text = _render(jobs=[])
        assert "Files processed:" in text
        assert "0" in text.split("Files processed:")[1].split("\n")[0]
        assert "Total files:" in text
        assert "0" in text.split("Total files:")[1].split("\n")[0]


class TestPerFileDisplay:
    def test_source_file_name_used(self) -> None:
        jobs = [
            make_job(
                status=JobStatus.DONE,
                source_files=["/movies/Avatar.mkv"],
                source_size=1_000_000,
                output_size=500_000,
            ),
        ]
        text = _render(jobs=jobs)
        assert "Avatar.mkv" in text

    def test_output_name_when_no_source_files(self) -> None:
        jobs = [
            make_job(
                status=JobStatus.DONE,
                source_files=[],
                output_file="/out/Fallback.mkv",
                source_size=1_000_000,
                output_size=500_000,
            ),
        ]
        text = _render(jobs=jobs)
        assert "Fallback.mkv" in text
