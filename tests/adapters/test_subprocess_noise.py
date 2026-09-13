from __future__ import annotations

import sys
from pathlib import Path

from furnace.adapters._subprocess import run_pipeline, run_tool

_DTS_LINE = (
    "[truehd @ 0000020beb8a5780] Application provided invalid, non monotonically "
    "increasing dts to muxer in stream 0: {n} >= {n}"
)


def _raw_spam(lines: list[str]) -> list[str]:
    """Only the verbatim ffmpeg complaints — the summary names the pattern too."""
    return [line for line in lines if "Application provided invalid" in line]


def _spammer(count: int, *, prefix: str = "", stream: str = "stderr") -> list[str]:
    """Child that prints `prefix`, then `count` ffmpeg muxer-DTS complaints."""
    code = (
        "import sys\n"
        f"out = sys.{stream}\n"
        f"pre = {prefix!r}\n"
        "if pre:\n"
        "    out.write(pre + '\\n')\n"
        f"for i in range({count}):\n"
        "    out.write('[truehd @ 0000020beb8a5780] Application provided invalid, "
        "non monotonically increasing dts to muxer in stream 0: %d >= %d\\n' % (i, i))\n"
        "out.flush()\n"
    )
    return [sys.executable, "-c", code]


class TestMuxerDtsNoise:
    def test_repeated_noise_collapses_to_first_line_plus_summary(self) -> None:
        received: list[str] = []
        rc, output = run_tool(_spammer(50), on_output=received.append)

        assert rc == 0
        spam = _raw_spam(received)
        assert len(spam) == 1, "only the first occurrence should reach the UI"
        assert "0 >= 0" in spam[0], "the surviving line must be the first one, verbatim"

        summaries = [line for line in received if "suppressed" in line]
        assert len(summaries) == 1
        assert "49" in summaries[0], "the summary must state how many were dropped"
        assert "non monotonically increasing dts to muxer" in summaries[0]
        assert summaries[0] in output

    def test_single_occurrence_needs_no_summary(self) -> None:
        received: list[str] = []
        rc, _output = run_tool(_spammer(1), on_output=received.append)

        assert rc == 0
        assert len(_raw_spam(received)) == 1
        assert not [line for line in received if "suppressed" in line]

    def test_unrelated_lines_are_never_touched(self) -> None:
        received: list[str] = []
        rc, output = run_tool(_spammer(30, prefix="Press [q] to stop"), on_output=received.append)

        assert rc == 0
        assert "Press [q] to stop" in received
        assert "Press [q] to stop" in output

    def test_summary_reaches_the_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "tool.log"
        rc, _output = run_tool(_spammer(40), log_path=log)

        assert rc == 0
        # the log opens with "$ <command>\n\n", and that command embeds the pattern
        body = log.read_text(encoding="utf-8").split("\n\n", 1)[1].split("\n")
        assert len(_raw_spam(body)) == 1
        assert any("suppressed 39" in line for line in body)
        assert any("exit code: 0" in line for line in body)

    def test_noise_is_dropped_from_returned_output(self) -> None:
        rc, output = run_tool(_spammer(200))

        assert rc == 0
        assert output.count("Application provided invalid") == 1

    def test_stdout_noise_is_suppressed_too(self) -> None:
        received: list[str] = []
        rc, _output = run_tool(_spammer(20, stream="stdout"), on_output=received.append)

        assert rc == 0
        assert len(_raw_spam(received)) == 1

    def test_progress_lines_take_precedence_over_noise_filter(self) -> None:
        seen: list[str] = []

        def parser(line: str) -> bool:
            seen.append(line)
            return "non monotonically" in line

        rc, output = run_tool(_spammer(10, prefix="ordinary output"), on_progress_line=parser)

        assert rc == 0
        assert len(seen) == 11, "the progress hook still sees every line"
        assert "ordinary output" in seen
        assert "ordinary output" in output
        assert "suppressed" not in output


class TestPipelineNoise:
    def test_pipeline_collapses_noise(self) -> None:
        received: list[str] = []
        rc, output = run_pipeline(
            [sys.executable, "-c", "pass"],
            _spammer(25),
            on_output=received.append,
        )

        assert rc == 0
        assert len(_raw_spam(received)) == 1
        assert "suppressed 24" in output


class TestChunkedReading:
    def test_line_longer_than_one_read_chunk_survives_intact(self) -> None:
        code = "import sys; sys.stdout.write('x' * 200000 + '\\ntail\\n'); sys.stdout.flush()"
        rc, output = run_tool([sys.executable, "-c", code])

        assert rc == 0
        lines = output.split("\n")
        assert "x" * 200000 in lines
        assert "tail" in lines

    def test_many_lines_keep_their_order_and_count(self) -> None:
        code = "import sys\nfor i in range(5000):\n    sys.stdout.write('line%d\\n' % i)\nsys.stdout.flush()\n"
        received: list[str] = []
        rc, _output = run_tool([sys.executable, "-c", code], on_output=received.append)

        assert rc == 0
        assert received == [f"line{i}" for i in range(5000)]

    def test_cr_and_lf_both_split_across_chunk_boundary(self) -> None:
        code = (
            "import sys\n"
            "sys.stdout.write('a' * 65535 + '\\r' + 'b' * 10 + '\\n')\n"
            "sys.stdout.flush()\n"
        )
        rc, output = run_tool([sys.executable, "-c", code])

        assert rc == 0
        lines = output.split("\n")
        assert "a" * 65535 in lines
        assert "b" * 10 in lines
