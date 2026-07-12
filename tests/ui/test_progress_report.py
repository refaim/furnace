"""Tests for furnace.ui.progress.ReportPrinter."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from furnace.core.models import JobStatus
from furnace.ui.progress import ReportPrinter
from tests.conftest import make_job, make_plan


def _render(*, vmaf_enabled: bool = False, **plan_kw: object) -> str:
    """Render a report and return captured text."""
    plan = make_plan(vmaf_enabled=vmaf_enabled, **plan_kw)  # type: ignore[arg-type]
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    ReportPrinter().print_report(plan, console)
    return buf.getvalue()


# ------------------------------------------------------------------
# 1. All jobs DONE
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 2. Error jobs
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 3. Mixed statuses
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 4. VMAF score threshold labels
# ------------------------------------------------------------------


class TestVmafLabels:
    def test_excellent_at_95(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=95.0)]
        text = _render(jobs=jobs)
        assert "VMAF 95.0 (excellent)" in text

    def test_excellent_above_95(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=98.7)]
        text = _render(jobs=jobs)
        assert "VMAF 98.7 (excellent)" in text

    def test_good_at_85(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=85.0)]
        text = _render(jobs=jobs)
        assert "VMAF 85.0 (good)" in text

    def test_good_at_94(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=94.9)]
        text = _render(jobs=jobs)
        assert "VMAF 94.9 (good)" in text

    def test_fair_at_70(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=70.0)]
        text = _render(jobs=jobs)
        assert "VMAF 70.0 (fair)" in text

    def test_fair_at_84(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=84.9)]
        text = _render(jobs=jobs)
        assert "VMAF 84.9 (fair)" in text

    def test_poor_below_70(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=69.9)]
        text = _render(jobs=jobs)
        assert "VMAF 69.9 (poor)" in text

    def test_poor_at_zero(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=0.0)]
        text = _render(jobs=jobs)
        assert "VMAF 0.0 (poor)" in text


# ------------------------------------------------------------------
# 5. SSIMULACRA2 / Butteraugli / CVVDP scores and labels
# ------------------------------------------------------------------


class TestSsimulacra2Labels:
    def test_excellent_at_90(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, ssimulacra2_score=90.0)]
        text = _render(jobs=jobs)
        assert "SSIMU2 90.00 (excellent)" in text

    def test_good_at_80(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, ssimulacra2_score=88.4)]
        text = _render(jobs=jobs)
        assert "SSIMU2 88.40 (good)" in text

    def test_fair_at_70(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, ssimulacra2_score=70.0)]
        text = _render(jobs=jobs)
        assert "SSIMU2 70.00 (fair)" in text

    def test_poor_below_70(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, ssimulacra2_score=55.5)]
        text = _render(jobs=jobs)
        assert "SSIMU2 55.50 (poor)" in text

    def test_poor_negative(self) -> None:
        """SSIMULACRA2 can be negative on very poor encodes."""
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, ssimulacra2_score=-18.1)]
        text = _render(jobs=jobs)
        assert "SSIMU2 -18.10 (poor)" in text


class TestButteraugliCvvdpDisplay:
    def test_butteraugli_value_shown(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, butteraugli_score=1.73)]
        text = _render(jobs=jobs)
        assert "Butteraugli 1.73" in text

    def test_cvvdp_value_shown(self) -> None:
        jobs = [make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, cvvdp_score=9.12)]
        text = _render(jobs=jobs)
        assert "CVVDP 9.12" in text


# ------------------------------------------------------------------
# 6. Average metric calculation
# ------------------------------------------------------------------


class TestAverages:
    def test_average_vmaf_shown_when_vmaf_enabled(self) -> None:
        jobs = [
            make_job(job_id="j1", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=90.0),
            make_job(job_id="j2", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=80.0),
        ]
        text = _render(vmaf_enabled=True, jobs=jobs)
        assert "Average:" in text
        assert "VMAF 85.00 (n=2)" in text

    def test_all_metric_averages_shown(self) -> None:
        jobs = [
            make_job(
                job_id="j1", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000,
                vmaf_score=90.0, ssimulacra2_score=88.0, butteraugli_score=1.5, cvvdp_score=9.0,
            ),
            make_job(
                job_id="j2", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000,
                vmaf_score=80.0, ssimulacra2_score=86.0, butteraugli_score=2.5, cvvdp_score=8.0,
            ),
        ]
        text = _render(vmaf_enabled=True, jobs=jobs)
        assert "VMAF 85.00 (n=2)" in text
        assert "SSIMU2 87.00 (n=2)" in text
        assert "Butteraugli 2.00 (n=2)" in text
        assert "CVVDP 8.50 (n=2)" in text

    def test_no_average_when_vmaf_disabled(self) -> None:
        jobs = [
            make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=90.0),
        ]
        text = _render(vmaf_enabled=False, jobs=jobs)
        assert "Average:" not in text

    def test_no_average_when_no_scores(self) -> None:
        jobs = [
            make_job(status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=None),
        ]
        text = _render(vmaf_enabled=True, jobs=jobs)
        assert "Average:" not in text

    def test_average_excludes_none_scores(self) -> None:
        jobs = [
            make_job(job_id="j1", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=90.0),
            make_job(job_id="j2", status=JobStatus.DONE, source_size=1_000_000, output_size=500_000, vmaf_score=None),
        ]
        text = _render(vmaf_enabled=True, jobs=jobs)
        assert "VMAF 90.00 (n=1)" in text

    def test_only_present_metrics_averaged(self) -> None:
        """Only metrics some job recorded appear; no separator for absent ones."""
        jobs = [
            make_job(
                job_id="j1", status=JobStatus.DONE, source_size=1_000_000,
                output_size=500_000, vmaf_score=90.0,
            ),
        ]
        text = _render(vmaf_enabled=True, jobs=jobs)
        assert "VMAF 90.00 (n=1)" in text
        # Only VMAF present -> no "  |  " separator to another metric.
        assert "  |  " not in text


# ------------------------------------------------------------------
# 7. Size savings
# ------------------------------------------------------------------


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
        # saved is negative -> sign is "+"
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
        # fmt_size(0) returns "?" and total_output == 0 is falsy
        assert "Total output size:" not in text
        assert "Space saved:" not in text


# ------------------------------------------------------------------
# 8. No done jobs
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# 9. Per-file display
# ------------------------------------------------------------------


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

    def test_all_metrics_on_same_line(self) -> None:
        jobs = [
            make_job(
                status=JobStatus.DONE,
                source_size=1_000_000,
                output_size=500_000,
                vmaf_score=92.5,
                ssimulacra2_score=87.5,
                butteraugli_score=1.9,
                cvvdp_score=9.3,
            ),
        ]
        text = _render(jobs=jobs)
        # All scores in the per-file section
        assert "VMAF 92.5 (good)" in text
        assert "SSIMU2 87.50 (good)" in text
        assert "Butteraugli 1.90" in text
        assert "CVVDP 9.30" in text

    def test_no_quality_when_scores_absent(self) -> None:
        jobs = [
            make_job(
                status=JobStatus.DONE,
                source_size=1_000_000,
                output_size=500_000,
                vmaf_score=None,
                ssimulacra2_score=None,
                butteraugli_score=None,
                cvvdp_score=None,
            ),
        ]
        text = _render(jobs=jobs)
        assert "VMAF" not in text
        assert "SSIMU2" not in text
        assert "Butteraugli" not in text
        assert "CVVDP" not in text
