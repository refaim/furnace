# dovi_tool extract-rpu via ffmpeg pipe — design

**Status:** approved
**Version target:** 1.14.5 (PATCH — bugfix; no behaviour change for non-DV jobs)

## Problem

`DoviToolAdapter.extract_rpu` passes the source MKV directly to
`dovi_tool extract-rpu`:

```python
cmd += ["extract-rpu", input_path, "-o", output_rpu]
```

`dovi_tool 2.3.1`'s `extract-rpu` accepts only an HEVC elementary stream
file or stdin (verbatim from `--help`: *"Sets the input HEVC file to use,
or piped with -"*). When given an MKV container it does **not** error —
it opens the output `RPU.bin`, then sits in a parser loop scanning
EBML/Matroska bytes for HEVC NAL units it will never find. Observed on a
real 85 GB Star-Trooper UHD remux: 15+ minutes of zero CPU progress,
RPU.bin remains 0 bytes, no log output, no failure surface.

Verified manually on the same source with the standard ffmpeg-pipe
approach:

```
ffmpeg -hide_banner -loglevel error -i <mkv> -map 0:v:0 -c copy \
  -bsf:v hevc_mp4toannexb -f hevc - \
  | dovi_tool -m 2 extract-rpu -l 1000 - -o RPU.bin
```

→ 1000 frames extracted in 1.5 s, 164 KB RPU. The MKV is fine, dovi_tool
is fine, the adapter is broken.

## Solution

Stop feeding MKVs to `dovi_tool`. Always pipe through ffmpeg, which
demuxes the container and emits a clean HEVC Annex B byte stream on its
stdout. dovi_tool reads that on its stdin (`-`).

Two pieces of work:

1. **New helper** `furnace.adapters._subprocess.run_pipeline(producer_cmd,
   consumer_cmd, …)` — a generic two-process Unix-pipeline runner that
   captures stderr from both ends and returns a combined exit code.
2. **Rewrite** `DoviToolAdapter` to build a producer (ffmpeg) command and
   a consumer (dovi_tool) command and dispatch them through
   `run_pipeline`. The public method signature
   (`extract_rpu(input_path, output_rpu, mode) -> int`) is unchanged so
   the executor and the `DoviProcessor` port stay untouched.

The adapter now needs the ffmpeg path; CLI wiring picks it up from the
already-loaded `cfg.ffmpeg`.

## Architecture

### `furnace/adapters/_subprocess.py` — new `run_pipeline`

```python
def run_pipeline(
    producer_cmd: Sequence[str | Path],
    consumer_cmd: Sequence[str | Path],
    on_output: OutputCallback = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> tuple[int, str]:
    """producer_cmd | consumer_cmd. Returns (rc, combined_stderr_text).

    rc = consumer_rc if consumer failed (≠0), else producer_rc — so any
    failure surfaces. Producer stdout is wired straight into consumer
    stdin (raw bytes, no decoding). Producer stderr, consumer stdout and
    consumer stderr are read on three threads and split on \\r or \\n
    using the same byte reader as run_tool. Decoded lines flow through
    on_output and into log_path.
    """
```

Key implementation points:

- `Popen(producer, stdin=DEVNULL, stdout=PIPE, stderr=PIPE)`,
  `Popen(consumer, stdin=producer.stdout, stdout=PIPE, stderr=PIPE)`.
- Parent **closes its copy of `producer.stdout` immediately** after the
  consumer is spawned. This is the standard CPython pattern; it lets the
  kernel deliver EOF / SIGPIPE correctly when the consumer exits before
  the producer is done.
- Three reader threads: `producer.stderr`, `consumer.stdout`,
  `consumer.stderr`. We do not read the consumer's stdout meaningfully
  for dovi_tool (it has none on `extract-rpu`) but we still drain it so
  a hypothetical future consumer that prints to stdout doesn't deadlock
  on a full pipe.
- Wait order: `consumer.wait()` first (it's the one whose exit we care
  about end-to-end), then `producer.wait()`.
- Log file format mirrors `run_tool` for consistency:
  ```
  $ <producer_cmd> | <consumer_cmd>

  <interleaved stderr / stdout lines>

  --- producer exit code: 0 ---
  --- consumer exit code: 0 ---
  ```
- No `cancel_event` in v1. The DV extract is the longest single step in
  the pipeline (3-30 min on UHD remuxes), but adding cancellation is a
  separate concern and the original adapter did not support it either.

Defensively NOT done:

- No fallback to the old "pass MKV directly" path. dovi_tool's silent
  hang IS the failure mode we're fixing; falling back would re-create it.
- No HEVC bitstream sniffing in the adapter. ffmpeg can read both raw
  Annex B `.hevc` and containered HEVC; `-bsf:v hevc_mp4toannexb` is a
  no-op when input is already Annex B. One code path, all inputs.

### `furnace/adapters/dovi_tool.py` — rewrite

Constructor gains a required `ffmpeg_path`:

```python
class DoviToolAdapter:
    def __init__(
        self,
        dovi_tool_path: Path,
        ffmpeg_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None: ...
```

Two private builders so each command is testable in isolation:

```python
def _build_ffmpeg_pipe_cmd(self, input_path: Path) -> list[str | Path]:
    return [
        self._ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-map", "0:v:0",
        "-c", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc",
        "-",
    ]

def _build_extract_cmd(
    self, output_rpu: Path, mode: DvMode,
) -> list[str | Path]:
    cmd: list[str | Path] = [self._dovi_tool]
    if mode == DvMode.TO_8_1:
        cmd += ["-m", "2"]
    cmd += ["extract-rpu", "-", "-o", output_rpu]
    return cmd
```

Note: `_build_extract_cmd` no longer takes `input_path` — the consumer
always reads stdin. The existing test cases that pass `Path("input.mkv")`
to `_build_extract_cmd` are rewritten in the plan.

`extract_rpu` becomes:

```python
def extract_rpu(
    self, input_path: Path, output_rpu: Path, mode: DvMode,
) -> int:
    producer = self._build_ffmpeg_pipe_cmd(input_path)
    consumer = self._build_extract_cmd(output_rpu, mode)
    log_path = (
        self._log_dir / "dovi_tool_extract.log" if self._log_dir else None
    )
    rc, _ = run_pipeline(
        producer, consumer,
        on_output=self._on_output, log_path=log_path,
    )
    return rc
```

Public API (`extract_rpu` signature, return-int-rc semantics) is
unchanged. `DoviProcessor` port in `core/ports.py` is **not** touched.

### `furnace/cli.py` — wiring

The existing block in `_run_executor`:

```python
if cfg.dovi_tool is not None:
    dovi_adapter = DoviToolAdapter(cfg.dovi_tool, on_output=tool_output)
```

becomes:

```python
if cfg.dovi_tool is not None:
    dovi_adapter = DoviToolAdapter(
        cfg.dovi_tool, cfg.ffmpeg, on_output=tool_output,
    )
```

`cfg.ffmpeg` is required by `ToolPaths` (already validated at config
load), so no None-check is needed.

### Executor / port — unchanged

`Executor._run_pipeline` calls
`self._dovi_processor.extract_rpu(input_path=main_source,
output_rpu=rpu_path, mode=...)` and branches on rc. Same as before.
`DoviProcessor` Protocol in `core/ports.py` is untouched.

## Data flow

Per DV job, in the per-job temp dir:

```
src.mkv
  └── ffmpeg | dovi_tool          (pipeline; HEVC Annex B in flight)
  └── RPU.bin                     ← consumer writes directly
```

Zero intermediate disk usage — the HEVC ES never lands on disk. This
matters: a 2-hour UHD HEVC track is ~50 GB.

## Error handling

`run_pipeline` returns the **first non-zero** of (consumer_rc,
producer_rc) so any failure on either side surfaces as a non-zero rc to
the executor, which already raises `RuntimeError(f"DV RPU extraction
failed with return code {rc}")`.

Specific failure cases that are now caught:

- ffmpeg can't open the input → ffmpeg rc≠0 → consumer reaches EOF early
  → small/empty RPU + non-zero rc.
- Source is not HEVC → ffmpeg `-c copy` fails on the bsf step → non-zero
  rc.
- Disk full while writing RPU → dovi_tool rc≠0 → non-zero rc.

Logs go to `<job>/dovi_tool_extract.log` containing both commands and
both exit codes plus interleaved stderr.

## Testing

Per CLAUDE.md: TDD strict, 100 % line + branch coverage. All test runs
through `make test` / `make check`.

### New test file: `tests/adapters/test_subprocess_pipeline.py`

Real subprocesses (the existing `test_subprocess_runner.py` pattern,
using `sys.executable -c "..."`):

- happy path: producer prints data → consumer reads via stdin → both
  rc 0 → output flows correctly
- producer rc != 0, consumer rc 0 → returned rc = producer's
- producer rc 0, consumer rc != 0 → returned rc = consumer's
- both rc != 0 → consumer's rc returned (consumer takes priority)
- producer stderr captured into on_output and log file
- consumer stderr captured into on_output and log file
- consumer stdout captured into on_output (it's drained, not silently
  buffered)
- producer's stdout bytes reach consumer (binary-clean — write a
  control-char payload and verify the consumer receives it byte-exact)
- log file contains both commands header and both exit code footers
- log_path None → no log file
- on_output None → no callback fires (smoke test)
- cwd is forwarded to both producer and consumer
- Popen pipe-validation guards (stdout/stderr None) raise — symmetric
  with the existing `TestRunToolPipeValidation` cases

### Updated test file: `tests/adapters/test_dovi_tool.py`

Drop or rewrite every test that built a `_build_extract_cmd(input_path,
…)`. New cases (using mocked `run_pipeline`):

- adapter constructor accepts `ffmpeg_path` positionally after
  `dovi_tool_path`; stored on the instance
- `_build_ffmpeg_pipe_cmd(Path("x.mkv"))` returns the expected ffmpeg
  argv (asserts `-bsf:v hevc_mp4toannexb`, `-f hevc`, trailing `-`,
  `-map 0:v:0`)
- `_build_extract_cmd(Path("RPU.bin"), DvMode.COPY)` → no `-m`, `-` for
  stdin, `-o` followed by output path
- `_build_extract_cmd(Path("RPU.bin"), DvMode.TO_8_1)` → `-m 2` present
- `extract_rpu` calls `run_pipeline` (not `run_tool`) with the producer
  cmd and consumer cmd produced by the two builders, plus the
  `dovi_tool_extract.log` log path when `log_dir` is set
- `extract_rpu` propagates the rc that `run_pipeline` returns (zero and
  non-zero cases)
- `set_log_dir(None)` clears the log path
- Default ctor (`log_dir=None`) → `log_path` arg to `run_pipeline` is
  `None`

### Updated test: `tests/test_cli.py::test_executor_fn_with_dovi_tool`

Add: `mock_dovi.assert_called_once_with(cfg.dovi_tool, cfg.ffmpeg,
on_output=...)`. The existing assertion (`assert_called_once`) is
extended, not replaced — the test still verifies the adapter is
constructed when `cfg.dovi_tool` is set.

### Untouched

`tests/services/test_executor.py` — every existing DoviProcessor test
mocks the port and asserts `extract_rpu(input_path=..., output_rpu=...,
mode=...)`. The signature is unchanged so all those tests still hold.

## Versioning

Bump to **1.14.5** in both `furnace/__init__.py` and `pyproject.toml`.
Rationale: this is a bugfix to a fully-broken feature — DV jobs now
succeed where they previously hung indefinitely. No new flag, no new
plan-JSON shape, no public-API change. PATCH per the project SemVer
rule.
