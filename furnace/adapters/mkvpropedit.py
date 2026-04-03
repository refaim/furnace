from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_TAGS_XML_TEMPLATE = """\
<Tags>
  <Tag>
    <Simple>
      <Name>ENCODER</Name>
      <String>{tag_value}</String>
    </Simple>
  </Tag>
</Tags>
"""


class MkvpropeditAdapter:
    """Implements Tagger. Sets ENCODER tag via mkvpropedit."""

    def __init__(self, mkvpropedit_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._mkvpropedit = mkvpropedit_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def set_encoder_tag(self, mkv_path: Path, tag_value: str) -> int:
        """Set global ENCODER tag.

        Creates a temporary tags.xml, runs:
            mkvpropedit mkv_path --tags global:tags.xml
        then deletes the temp file.
        """
        xml_content = _TAGS_XML_TEMPLATE.format(tag_value=tag_value)

        # Write temp file in the same directory as the MKV for locality
        tmp_dir = mkv_path.parent
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(tmp_dir), suffix=".xml", prefix="furnace_tags_"
        )
        tmp_path = Path(tmp_path_str)
        try:
            import os
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(xml_content)

            cmd = [
                str(self._mkvpropedit),
                str(mkv_path),
                "--tags", f"global:{tmp_path}",
            ]
            log_path = self._log_dir / "mkvpropedit.log" if self._log_dir else None
            rc, _stderr = run_tool(cmd, on_output=self._on_output, log_path=log_path)
            return rc
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
