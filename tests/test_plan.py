from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from furnace.core.models import (
    AudioAction,
    AudioInstruction,
    ColorSpace,
    CropRect,
    HdrMetadata,
    Job,
    JobStatus,
    Plan,
    SubtitleAction,
    SubtitleInstruction,
    VideoParams,
)
from furnace.plan import atomic_write, load_plan, save_plan, update_job_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_video_params(
    cq: int = 25,
    crop: CropRect | None = None,
    deinterlace: bool = False,
    color_space: ColorSpace = ColorSpace.BT709,
    color_range: str = "tv",
    color_transfer: str | None = None,
    color_primaries: str | None = None,
    hdr: HdrMetadata | None = None,
    gop: int = 120,
    fps_num: int = 24,
    fps_den: int = 1,
    source_width: int = 1920,
    source_height: int = 1080,
) -> VideoParams:
    return VideoParams(
        cq=cq,
        crop=crop,
        deinterlace=deinterlace,
        color_space=color_space,
        color_range=color_range,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        hdr=hdr,
        gop=gop,
        fps_num=fps_num,
        fps_den=fps_den,
        source_width=source_width,
        source_height=source_height,
    )


def make_audio_instruction(
    source_file: str = "/src/movie.mkv",
    stream_index: int = 1,
    language: str = "eng",
    action: AudioAction = AudioAction.COPY,
    delay_ms: int = 0,
    is_default: bool = True,
    codec_name: str = "aac",
    channels: int | None = 2,
    bitrate: int | None = 192000,
) -> AudioInstruction:
    return AudioInstruction(
        source_file=source_file,
        stream_index=stream_index,
        language=language,
        action=action,
        delay_ms=delay_ms,
        is_default=is_default,
        codec_name=codec_name,
        channels=channels,
        bitrate=bitrate,
    )


def make_subtitle_instruction(
    source_file: str = "/src/movie.mkv",
    stream_index: int = 2,
    language: str = "eng",
    action: SubtitleAction = SubtitleAction.COPY,
    is_default: bool = False,
    is_forced: bool = False,
    codec_name: str = "hdmv_pgs_subtitle",
    source_encoding: str | None = None,
) -> SubtitleInstruction:
    return SubtitleInstruction(
        source_file=source_file,
        stream_index=stream_index,
        language=language,
        action=action,
        is_default=is_default,
        is_forced=is_forced,
        codec_name=codec_name,
        source_encoding=source_encoding,
    )


def make_job(
    job_id: str = "test-job-001",
    output_file: str = "/out/movie.mkv",
    status: JobStatus = JobStatus.PENDING,
    source_files: list[str] | None = None,
    audio: list[AudioInstruction] | None = None,
    subtitles: list[SubtitleInstruction] | None = None,
) -> Job:
    return Job(
        id=job_id,
        source_files=["/src/movie.mkv"] if source_files is None else source_files,
        output_file=output_file,
        video_params=make_video_params(),
        audio=[make_audio_instruction()] if audio is None else audio,
        subtitles=[make_subtitle_instruction()] if subtitles is None else subtitles,
        attachments=[],
        copy_chapters=True,
        chapters_source=None,
        status=status,
        error=None,
        vmaf_score=None,
        source_size=1_000_000,
        output_size=None,
    )


def make_plan(jobs: list[Job] | None = None, demux_dir: str | None = None) -> Plan:
    return Plan(
        version="2",
        furnace_version="0.1.0",
        created_at="2026-04-01T00:00:00",
        source="/src",
        destination="/out",
        vmaf_enabled=False,
        demux_dir=demux_dir,
        jobs=[make_job()] if jobs is None else jobs,
    )


# ---------------------------------------------------------------------------
# test_plan_roundtrip
# ---------------------------------------------------------------------------

class TestPlanRoundtrip:
    def test_basic_roundtrip(self, tmp_path):
        """save -> load -> save produces identical JSON."""
        plan = make_plan()
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        json1 = plan_path.read_text(encoding="utf-8")

        loaded = load_plan(plan_path)
        save_plan(loaded, plan_path)
        json2 = plan_path.read_text(encoding="utf-8")

        assert json1 == json2

    def test_roundtrip_with_crop(self, tmp_path):
        """Plan with CropRect survives roundtrip."""
        crop = CropRect(w=1920, h=800, x=0, y=140)
        vp = make_video_params(crop=crop)
        job = Job(
            id="crop-job",
            source_files=["/src/movie.mkv"],
            output_file="/out/movie.mkv",
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            chapters_source=None,
            status=JobStatus.PENDING,
            source_size=0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert loaded.jobs[0].video_params.crop == crop

    def test_roundtrip_with_hdr(self, tmp_path):
        """Plan with HDR metadata survives roundtrip."""
        hdr = HdrMetadata(
            mastering_display="G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(1000,0.005)",
            content_light="MaxCLL=1000,MaxFALL=400",
            is_dolby_vision=False,
            is_hdr10_plus=False,
        )
        vp = make_video_params(
            color_space=ColorSpace.BT2020,
            color_transfer="smpte2084",
            color_primaries="bt2020",
            hdr=hdr,
        )
        job = Job(
            id="hdr-job",
            source_files=["/src/hdr.mkv"],
            output_file="/out/hdr.mkv",
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            chapters_source=None,
            status=JobStatus.PENDING,
            source_size=0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        loaded_hdr = loaded.jobs[0].video_params.hdr
        assert loaded_hdr is not None
        assert loaded_hdr.mastering_display == hdr.mastering_display
        assert loaded_hdr.content_light == hdr.content_light

    def test_roundtrip_preserves_job_fields(self, tmp_path):
        """All job scalar fields survive roundtrip."""
        job = make_job(job_id="abc-123", output_file="/out/test.mkv")
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        loaded_job = loaded.jobs[0]
        assert loaded_job.id == "abc-123"
        assert loaded_job.output_file == "/out/test.mkv"
        assert loaded_job.status == JobStatus.PENDING

    def test_roundtrip_empty_jobs(self, tmp_path):
        """Plan with no jobs roundtrips correctly."""
        plan = make_plan(jobs=[])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert loaded.jobs == []

    def test_roundtrip_multiple_jobs(self, tmp_path):
        """Multiple jobs all survive roundtrip."""
        jobs = [make_job(job_id=f"job-{i}", output_file=f"/out/movie{i}.mkv") for i in range(3)]
        plan = make_plan(jobs=jobs)
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert len(loaded.jobs) == 3
        assert [j.id for j in loaded.jobs] == ["job-0", "job-1", "job-2"]

    def test_roundtrip_audio_action_preserved(self, tmp_path):
        """AudioAction enum values survive roundtrip."""
        audio = make_audio_instruction(action=AudioAction.DECODE_ENCODE)
        job = make_job(audio=[audio])
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert loaded.jobs[0].audio[0].action == AudioAction.DECODE_ENCODE

    def test_roundtrip_subtitle_instruction(self, tmp_path):
        """SubtitleInstruction with COPY_RECODE survives roundtrip."""
        sub = make_subtitle_instruction(
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            source_encoding="cp1251",
            is_forced=True,
        )
        job = make_job(subtitles=[sub])
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        loaded_sub = loaded.jobs[0].subtitles[0]
        assert loaded_sub.action == SubtitleAction.COPY_RECODE
        assert loaded_sub.source_encoding == "cp1251"
        assert loaded_sub.is_forced is True


# ---------------------------------------------------------------------------
# test_plan_version_validation
# ---------------------------------------------------------------------------

class TestPlanVersionValidation:
    def test_correct_version_loads(self, tmp_path):
        """Plan with version '2' loads without error."""
        plan = make_plan()
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        assert loaded.version == "2"

    def test_wrong_version_raises(self, tmp_path):
        """Plan with wrong version -> ValueError."""
        plan_path = tmp_path / "plan.json"
        data = {"version": "99", "furnace_version": "0.1.0", "created_at": "2026-01-01T00:00:00",
                "source": "/src", "destination": "/out", "vmaf_enabled": False, "jobs": []}
        plan_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported plan version"):
            load_plan(plan_path)

    def test_missing_version_raises(self, tmp_path):
        """Plan with no version field -> ValueError."""
        plan_path = tmp_path / "plan.json"
        data = {"furnace_version": "0.1.0", "created_at": "2026-01-01T00:00:00",
                "source": "/src", "destination": "/out", "vmaf_enabled": False, "jobs": []}
        plan_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported plan version"):
            load_plan(plan_path)


# ---------------------------------------------------------------------------
# test_update_job_status
# ---------------------------------------------------------------------------

class TestUpdateJobStatus:
    def test_update_pending_to_done(self, tmp_path):
        """Update job from PENDING -> DONE."""
        plan = make_plan(jobs=[make_job(job_id="j1")])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        update_job_status(plan_path, "j1", JobStatus.DONE)

        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        job_raw = raw["jobs"][0]
        assert job_raw["status"] == "done"
        assert job_raw["error"] is None

    def test_update_to_error_with_message(self, tmp_path):
        """Update job to ERROR with error message."""
        plan = make_plan(jobs=[make_job(job_id="j2")])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        update_job_status(plan_path, "j2", JobStatus.ERROR, error="ffmpeg died")

        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        job_raw = raw["jobs"][0]
        assert job_raw["status"] == "error"
        assert job_raw["error"] == "ffmpeg died"

    def test_update_with_vmaf_score(self, tmp_path):
        """vmaf_score is persisted when provided."""
        plan = make_plan(jobs=[make_job(job_id="j3")])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        update_job_status(plan_path, "j3", JobStatus.DONE, vmaf_score=95.4)

        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        assert raw["jobs"][0]["vmaf_score"] == pytest.approx(95.4)

    def test_update_with_output_size(self, tmp_path):
        """output_size is persisted when provided."""
        plan = make_plan(jobs=[make_job(job_id="j4")])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        update_job_status(plan_path, "j4", JobStatus.DONE, output_size=500_000_000)

        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        assert raw["jobs"][0]["output_size"] == 500_000_000

    def test_update_nonexistent_job_raises(self, tmp_path):
        """Updating a job ID that doesn't exist -> KeyError."""
        plan = make_plan(jobs=[make_job(job_id="j5")])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        with pytest.raises(KeyError):
            update_job_status(plan_path, "nonexistent", JobStatus.DONE)

    def test_update_correct_job_among_multiple(self, tmp_path):
        """Only the targeted job is updated when multiple jobs exist."""
        jobs = [make_job(job_id="j-a"), make_job(job_id="j-b"), make_job(job_id="j-c")]
        plan = make_plan(jobs=jobs)
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        update_job_status(plan_path, "j-b", JobStatus.DONE)

        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        statuses = {j["id"]: j["status"] for j in raw["jobs"]}
        assert statuses["j-a"] == "pending"
        assert statuses["j-b"] == "done"
        assert statuses["j-c"] == "pending"


# ---------------------------------------------------------------------------
# test_atomic_write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        """atomic_write creates the file with correct content."""
        target = tmp_path / "output.json"
        atomic_write(target, '{"key": "value"}')
        assert target.exists()
        assert target.read_text(encoding="utf-8") == '{"key": "value"}'

    def test_overwrites_existing(self, tmp_path):
        """atomic_write overwrites an existing file atomically."""
        target = tmp_path / "output.json"
        target.write_text("old content", encoding="utf-8")
        atomic_write(target, "new content")
        assert target.read_text(encoding="utf-8") == "new content"

    def test_no_tmp_file_left_on_success(self, tmp_path):
        """No .tmp file remains after successful write."""
        target = tmp_path / "output.json"
        atomic_write(target, "data")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_unicode_content(self, tmp_path):
        """atomic_write handles Unicode content correctly."""
        target = tmp_path / "output.json"
        content = '{"title": "Фильм — тест"}'
        atomic_write(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_write_to_nested_existing_dir(self, tmp_path):
        """atomic_write works when parent dir exists."""
        subdir = tmp_path / "plans"
        subdir.mkdir()
        target = subdir / "plan.json"
        atomic_write(target, "content")
        assert target.read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# test_plan_demux_dir
# ---------------------------------------------------------------------------

class TestPlanDemuxDir:
    def test_roundtrip_with_demux_dir(self, tmp_path):
        plan = make_plan(demux_dir="/src/.furnace_demux")
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        assert loaded.demux_dir == "/src/.furnace_demux"

    def test_roundtrip_without_demux_dir(self, tmp_path):
        plan = make_plan(demux_dir=None)
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        assert loaded.demux_dir is None
