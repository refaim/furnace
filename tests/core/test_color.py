from __future__ import annotations

import pytest

from furnace.core.color import (
    CICP_MATRIX,
    CICP_PRIMARIES,
    CICP_TRANSFER,
    MasteringDisplay,
    parse_content_light,
    parse_mastering_display,
)

_UHD_1000_NITS = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)"


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


class TestParseMasteringDisplay:
    def test_bt2020_primaries(self) -> None:
        md = parse_mastering_display(_UHD_1000_NITS)
        assert md.red == (0.68, 0.32)
        assert md.green == (0.265, 0.69)
        assert md.blue == (0.15, 0.06)

    def test_d65_white_point(self) -> None:
        md = parse_mastering_display(_UHD_1000_NITS)
        assert md.white == (0.3127, 0.329)

    def test_luminance_scaled_to_nits(self) -> None:
        md = parse_mastering_display(_UHD_1000_NITS)
        assert md.max_luminance == 1000.0
        assert md.min_luminance == 0.0

    def test_fractional_min_luminance(self) -> None:
        md = parse_mastering_display(
            "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(20000000,1)"
        )
        assert md.max_luminance == 2000.0
        assert md.min_luminance == 0.0001

    def test_returns_frozen_dataclass(self) -> None:
        md = parse_mastering_display(_UHD_1000_NITS)
        assert isinstance(md, MasteringDisplay)
        with pytest.raises(AttributeError):
            md.max_luminance = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "not a mastering display",
            "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)",
            "G(13250,34500)B(7500,3000)R(34000,16000)L(10000000,0)",
            "G(a,b)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)",
            "G(13250,34500) B(7500,3000) R(34000,16000) WP(15635,16450) L(10000000,0)",
        ],
    )
    def test_malformed_raises(self, value: str) -> None:
        with pytest.raises(ValueError, match="mastering display"):
            parse_mastering_display(value)


class TestParseContentLight:
    def test_valid_input(self) -> None:
        assert parse_content_light("MaxCLL=1000,MaxFALL=400") == ("1000", "400")

    def test_valid_with_spaces(self) -> None:
        assert parse_content_light("MaxCLL=1000, MaxFALL=400") == ("1000", "400")

    def test_zero_levels_parse(self) -> None:
        assert parse_content_light("MaxCLL=0,MaxFALL=0") == ("0", "0")

    def test_invalid_input(self) -> None:
        assert parse_content_light("not valid") is None

    def test_empty_levels_do_not_parse(self) -> None:
        assert parse_content_light("MaxCLL=,MaxFALL=147") is None


class TestConsistency:
    def test_all_codes_in_cicp_byte_range(self) -> None:
        for table in (CICP_PRIMARIES, CICP_TRANSFER, CICP_MATRIX):
            for code in table.values():
                assert 0 <= code <= 255

    def test_pal_and_ntsc_matrix_share_bt601_coefficients(self) -> None:
        assert "bt470bg" in CICP_MATRIX
        assert "smpte170m" in CICP_MATRIX
