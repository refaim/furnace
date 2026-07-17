from __future__ import annotations

import pytest

from furnace.core.color import CICP_MATRIX, CICP_PRIMARIES, CICP_TRANSFER


class TestCicpPrimaries:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("bt709", 1),
            ("bt470m", 4),
            ("bt470bg", 5),
            ("smpte170m", 6),
            ("smpte240m", 7),
            ("bt2020", 9),
        ],
    )
    def test_values(self, name: str, code: int) -> None:
        assert CICP_PRIMARIES[name] == code


class TestCicpTransfer:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("bt709", 1),
            ("bt470m", 4),
            ("bt470bg", 5),
            ("smpte170m", 6),
            ("smpte240m", 7),
            ("linear", 8),
            ("bt2020-10", 14),
            ("bt2020-12", 15),
            ("smpte2084", 16),
            ("arib-std-b67", 18),
        ],
    )
    def test_values(self, name: str, code: int) -> None:
        assert CICP_TRANSFER[name] == code


class TestCicpMatrix:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("bt709", 1),
            ("bt470bg", 5),
            ("smpte170m", 6),
            ("smpte240m", 7),
            ("bt2020nc", 9),
            ("bt2020c", 10),
        ],
    )
    def test_values(self, name: str, code: int) -> None:
        assert CICP_MATRIX[name] == code


class TestConsistency:
    def test_all_codes_in_cicp_byte_range(self) -> None:
        for table in (CICP_PRIMARIES, CICP_TRANSFER, CICP_MATRIX):
            for code in table.values():
                assert 0 <= code <= 255

    def test_pal_and_ntsc_matrix_share_bt601_coefficients(self) -> None:
        assert "bt470bg" in CICP_MATRIX
        assert "smpte170m" in CICP_MATRIX
