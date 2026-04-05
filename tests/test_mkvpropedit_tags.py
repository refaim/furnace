"""Tests for mkvpropedit _build_tags_xml helper."""
from __future__ import annotations

from furnace.adapters.mkvpropedit import _build_tags_xml


class TestBuildTagsXml:
    def test_encoder_only(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0")
        assert "<Name>ENCODER</Name>" in xml
        assert "<String>Furnace v1.4.0</String>" in xml
        assert "ENCODER_SETTINGS" not in xml

    def test_with_encoder_settings(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0", "hevc_nvenc / main10 / cq=25")
        assert "<Name>ENCODER</Name>" in xml
        assert "<Name>ENCODER_SETTINGS</Name>" in xml
        assert "<String>hevc_nvenc / main10 / cq=25</String>" in xml
