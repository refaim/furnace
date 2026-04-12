from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters._subprocess import run_tool


class TestRunToolPipeValidation:
    def test_raises_when_stdout_missing(self) -> None:
        mock_process = MagicMock()
        mock_process.stdout = None
        mock_process.stderr = MagicMock()
        with patch("furnace.adapters._subprocess.subprocess.Popen", return_value=mock_process), \
             pytest.raises(RuntimeError, match="pipes"):
            run_tool(["echo", "x"])

    def test_raises_when_stderr_missing(self) -> None:
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = None
        with patch("furnace.adapters._subprocess.subprocess.Popen", return_value=mock_process), \
             pytest.raises(RuntimeError, match="pipes"):
            run_tool(["echo", "x"])
