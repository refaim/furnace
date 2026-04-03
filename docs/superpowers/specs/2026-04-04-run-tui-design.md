# Run TUI Design — Textual-based encoding progress display

## Problem

Rich Live + console logging handler conflict in cmd.exe: logging output breaks Rich Live cursor positioning, causing header duplication and display artifacts. No clean way to show both pipeline logs and progress bar.

## Solution

Replace Rich Live with a Textual App for the `furnace run` phase. Textual owns the terminal entirely, eliminating logging conflicts. ASCII-only borders for cmd.exe compatibility.

## Layout

```
+-- [1/3] Movie Name (2020) ------------------------------------+
|                                                                |
+-- Source -------------------------+-- Target -----------------+
| Video: H.264 1920x1080 8.5Mbps   | Video: HEVC 1920x800 CQ25|
| Audio: DTS 5.1 755kbps           | Audio: DTS 5.1 (denorm)   |
|        AC3 2.0 192kbps           |        AC3 2.0 (denorm)   |
| Subs:  SRT rus                    | Subs:  SRT rus (UTF-8)   |
| Size:  4.0 GB                     |                           |
+-----------------------------------+---------------------------+
|                                                                |
+-- Steps ------+-- Output -------------------------------------+
|   Extract     |  Running in normal mode                        |
| > Denormalize |  Removing dialnorm                             |
|   Encode      |  DTS, 5.1, 1:41:56, 755kbps, 48kHz            |
|   Mux         |  Creating file "audio_2_denorm.dts"...         |
|   Optimize    |  eac3to processing took 4 seconds.             |
|               |  Done.                                         |
+---------------+------------------------------------------------+
|                                                                |
+-- ████████████░░░░░░░░░░  42.5% | 3:20 / ~4:40 | 1.2x -------+
+----------------------------------------------------------------+
```

### Zones

1. **Header** (1 line, top): `[X/N] Output filename` — batch progress and current file
2. **Source/Target** (side-by-side, ~6 lines): What we have vs what we're making
3. **Steps + Output** (side-by-side, fills remaining height):
   - Steps (fixed ~15 char width): pipeline step list with current step marker
   - Output (fills rest): scrollable tool stdout/stderr
4. **Progress bar** (1-2 lines, bottom): Unicode bar + pct + elapsed/ETA + speed

### Responsive

All zones stretch to terminal width. Textual CSS handles this automatically:
- Source/Target: 50%/50% width
- Steps: fixed width, Output: `1fr` (remaining)
- Output: scrollable vertically

## Zones Detail

### Header

Static widget, updated per-job:
```
[1/3] Movie Name (2020)
```

### Source block

From `Job` data — shows what the source file contains:
```
Video: H.264 1920x1080 8.5Mbps
Audio: DTS 5.1 755kbps
       AC3 2.0 192kbps
Subs:  SRT rus
Size:  4.0 GB
```

Video bitrate from ffprobe `format.bit_rate` or stream `bit_rate`.

### Target block

From `Job.video_params` + audio/subtitle instructions — shows what we're encoding to:
```
Video: HEVC 1920x800 CQ25
Audio: DTS 5.1 (denorm)
       AC3 2.0 (denorm)
Subs:  SRT rus (UTF-8)
```

Resolution is crop dimensions if crop is set, otherwise source dimensions.

Audio target format depends on AudioAction:
- COPY: codec + channels (copy)
- DENORM: codec + channels (denorm)
- DECODE_ENCODE: AAC (from codec)
- FFMPEG_ENCODE: AAC (from codec)

### Steps block

Static list of pipeline steps. Current step has `>` marker, completed steps have `+`:
```
+ Extract
+ Denormalize
> Encode
  Mux
  Optimize
```

Human-readable step names:
| Internal | Display |
|----------|---------|
| extract_track | Extract |
| denormalize | Denormalize |
| decode_lossless | Decode |
| encode_aac | Encode AAC |
| encode (ffmpeg) | Encode video |
| mux | Mux |
| set_encoder_tag | Tag |
| mkclean | Optimize |

Steps list is dynamic per job — only shows steps that apply. E.g., if audio is COPY, no Denormalize step. If no subs, no subtitle steps.

### Output block

Scrollable container showing tool stdout+stderr in real-time. New lines auto-scroll to bottom. Keeps full history per-job (cleared on next job).

Lines matching `process: XX%` (eac3to progress) are filtered out and routed to the progress bar instead.

### Progress bar

Unicode blocks (crucible-style), updated via callback:
```
████████████░░░░░░░░░░  42.5% | 3:20 / ~4:40 | 1.2x
```

Shows:
- Bar with percentage
- Elapsed time / estimated remaining
- Encoding speed (from ffmpeg)

For non-encoding steps: indeterminate state or hidden.
For eac3to `process: XX%`: shows eac3to progress.

## Interaction

- **ESC**: Graceful shutdown (Textual binding). Sets shutdown event, kills ffmpeg tree via psutil.
- **No other keyboard interaction needed** during encoding.
- App exits automatically when all jobs complete.

## Architecture

### New file: `furnace/ui/run_tui.py`

```python
class RunApp(App):
    """Textual app for furnace run phase."""
    
    # Widgets:
    # - HeaderWidget (Static): "[X/N] filename"
    # - SourceWidget (Static): source track info
    # - TargetWidget (Static): target track info  
    # - StepsWidget (Static): pipeline step list
    # - OutputWidget (RichLog or ScrollableContainer): tool output
    # - ProgressWidget (Static): progress bar

    # CSS: all borders ascii, responsive layout
```

### Integration

`cli.py` `run()` command:
1. Creates `RunApp` instead of `EncodingProgress`
2. Runs `RunApp` — it takes over the terminal
3. Executor callbacks go to RunApp methods (same interface as EncodingProgress)
4. After app exits, print report to console (outside Textual)

### Callback interface (same as current EncodingProgress)

```python
start_job(job, job_index)     # update header + source/target + reset steps
update_encode(pct, speed)     # update progress bar
update_status(message)        # update current step in steps list
add_tool_line(line)           # append to output widget
finish_job(job)               # mark all steps done
stop()                        # exit app
```

### Logging during TUI

Console logging handler is OFF (Textual owns terminal). File logging continues to `furnace.log`. Pipeline step `logger.info()` calls are paired with `add_tool_line()` calls from executor so they appear in both the Output widget and the log file.

## CSS

All borders `ascii` for cmd.exe compatibility. No Unicode box-drawing characters.

```css
#header { height: 1; }
#source-target { height: auto; max-height: 10; }
#source { width: 50%; border: ascii; }
#target { width: 50%; border: ascii; }
#steps { width: 18; border: ascii; }
#output { border: ascii; }  /* fills remaining */
#progress { height: 2; border: ascii; }
```

## Non-goals

- No mouse interaction
- No color themes (use Textual defaults with ascii borders)
- No pause/resume (only ESC to stop)
- No per-file report in TUI (report prints after TUI exits)
