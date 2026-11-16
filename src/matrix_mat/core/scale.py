"""
Scale & Metadata Extraction Module for Brightfield Microscopy Images.
Extracts optical pixel scale (um/pixel) and microscope metadata from TIFF tags
(Tag 37510 EVOS JSON, Tag 270 OME-XML, ImageJ metadata, and TIFF resolution tags)
with robust CLI override and fallback handling.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload
from PIL import Image
import tifffile

logger = logging.getLogger("matrix_mat.core.scale")


PLAUSIBLE_UM_PER_PX = (0.05, 20.0)
# OME-XML UnitsLength symbols -> micrometres
OME_LENGTH_TO_UM = {"µm": 1.0, "um": 1.0, "\u03bcm": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6, "pm": 1e-6}
# ImageJ / Fiji calibration unit names -> micrometres (None = explicitly uncalibrated)
IMAGEJ_UNIT_TO_UM: Dict[str, Optional[float]] = {
    "um": 1.0, "µm": 1.0, "\u03bcm": 1.0, "micron": 1.0, "microns": 1.0, "micrometer": 1.0,
    "micrometre": 1.0, "micrometers": 1.0, "micrometres": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3, "nanometers": 1e-3,
    "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3, "millimeters": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4, "centimeters": 1e4,
    "m": 1e6, "meter": 1e6, "metre": 1e6, "meters": 1e6,
    "inch": 25400.0, "inches": 25400.0, "in": 25400.0,
    "pixel": None, "pixels": None, "px": None, "": None,
}


def _decode_text(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return val if isinstance(val, str) else str(val)


def _normalise_unit(unit: str) -> str:
    u = unit.strip()
    # ImageJ escapes the micro sign as a literal \u00b5 in the description text.
    u = re.sub(r"\\u00[bB]5", "\u00b5", u)
    return u.replace("\u03bc", "\u00b5").lower()


def _imagej_unit_factor(unit: str) -> Tuple[bool, Optional[float]]:
    """Return (known, factor_to_um). factor is None for an explicitly uncalibrated unit (pixel)."""
    u = _normalise_unit(unit)
    if u in IMAGEJ_UNIT_TO_UM:
        return True, IMAGEJ_UNIT_TO_UM[u]
    return False, None


def _lenient_float(val: Any) -> Optional[float]:
    """Parse numbers such as 4, "4.0", "4x", "40X", " 0.13 NA"; None if no number is present."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _is_plausible(px: float) -> bool:
    return PLAUSIBLE_UM_PER_PX[0] <= px <= PLAUSIBLE_UM_PER_PX[1]


@dataclass
class ScaleInfo:
    """Metadata extracted for optical scaling and calibration."""
    pixel_size_um: float
    source: str
    magnification: Optional[float] = None
    numerical_aperture: Optional[float] = None
    instrument_model: Optional[str] = None
    is_calibrated: bool = True
    warning: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class ScaleExtractor:
    """Multi-strategy scale and metadata extractor for microscopy images."""

    def __init__(self, default_pixel_size_um: float = 1.518817, cli_override_um: Optional[float] = None):
        self.default_pixel_size_um = float(default_pixel_size_um)
        self.cli_override_um = float(cli_override_um) if cli_override_um is not None else None

    def _rejected(self, reason_source: str, warning: str, **kw: Any) -> ScaleInfo:
        """A metadata candidate that must not be trusted.

        The configured default pixel size is returned (never the rejected value), with
        ``is_calibrated=False`` and a ``source`` naming why the embedded value was rejected.
        """
        logger.warning(warning)
        return ScaleInfo(
            pixel_size_um=self.default_pixel_size_um,
            source=f"config_default_uncalibrated ({reason_source})",
            is_calibrated=False,
            warning=warning,
            **kw,
        )

    def extract(self, image_path: str | Path) -> ScaleInfo:
        """
        Extract pixel size (um/pixel) and metadata from an image file.

        Priority order:
        1. CLI override if specified (--pixel-size)
        2. Embedded EVOS JSON (TIFF Tag 37510 or Tag 270 comment)
        3. OME-XML (Tag 270 PhysicalSizeX / PhysicalSizeY)
        4. ImageJ metadata (pixelWidth, or XResolution in the ImageJ ``unit``)
        5. Standard TIFF resolution tags (Tags 282, 283, 296)
        6. Default config fallback (with warning log)

        Candidates that are found but rejected (implausible values, 72/96 DPI screen
        defaults) never contribute their pixel size: the result is the configured default
        with ``is_calibrated=False`` and the rejection reason in ``source``/``warning``.
        """
        image_path = Path(image_path)

        # Strategy 1: CLI override
        if self.cli_override_um is not None:
            logger.info(f"Using CLI pixel size override: {self.cli_override_um} um/pixel for {image_path.name}")
            return ScaleInfo(
                pixel_size_um=self.cli_override_um,
                source="cli_override",
                is_calibrated=True,
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        rejected: Optional[ScaleInfo] = None

        def _strategies(tags: Dict[int, Any], info_dict: Dict[str, Any], size: Tuple[int, int]):
            yield self._extract_evos_metadata(tags)
            yield self._extract_ome_xml(tags, size)
            yield self._extract_imagej(tags, info_dict)
            yield self._extract_tiff_resolution(tags, info_dict)

        # Attempt extraction via tifffile first (most accurate for scientific TIFFs)
        try:
            with tifffile.TiffFile(image_path) as tif:
                for page in tif.pages:
                    if not isinstance(page, tifffile.TiffPage):
                        continue
                    tags = {tag.code: tag.value for tag in page.tags.values()}
                    size = (page.shape[1], page.shape[0]) if len(page.shape) >= 2 else (2048, 1536)
                    for info in _strategies(tags, {}, size):
                        if info is None:
                            continue
                        if info.is_calibrated:
                            return info
                        rejected = rejected or info
                    if rejected is not None:
                        return rejected
        except Exception as e:
            logger.debug(f"tifffile extraction encounter: {e}, attempting PIL fallback...")

        # Fallback to PIL inspection if tifffile did not resolve
        try:
            with Image.open(image_path) as im:
                tags = dict(getattr(im, "tag_v2", {}) or {})
                info_dict = getattr(im, "info", {})
                for info in _strategies(tags, info_dict, im.size):
                    if info is None:
                        continue
                    if info.is_calibrated:
                        return info
                    rejected = rejected or info
        except Exception as e:
            logger.warning(f"Error reading metadata from {image_path.name}: {e}")

        if rejected is not None:
            return rejected

        # Fallback Strategy: Default uncalibrated config
        warn_msg = (
            f"WARNING: No embedded physical scale metadata found in {image_path.name}. "
            f"Using default of {self.default_pixel_size_um} um/pixel."
        )
        logger.warning(warn_msg)
        return ScaleInfo(
            pixel_size_um=self.default_pixel_size_um,
            source="config_default_uncalibrated",
            is_calibrated=False,
            warning=warn_msg,
        )

    def _extract_evos_metadata(self, tags: Dict[int, Any]) -> Optional[ScaleInfo]:
        """Extract metadata from ThermoFisher EVOS specific tags (Tag 37510 or JSON comments)."""
        json_data: Optional[Dict[str, Any]] = None

        # Check Tag 37510
        if 37510 in tags:
            val = _decode_text(tags[37510])
            m = re.search(r"(\{.*\})", val, re.DOTALL)
            if m:
                try:
                    json_data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Check Tag 270 comment block for EVOS JSON
        if json_data is None and 270 in tags:
            desc = _decode_text(tags[270])
            m = re.search(r"<!--\s*(\{.*?\})\s*-->", desc, re.DOTALL)
            if m:
                try:
                    json_data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if not json_data or not isinstance(json_data, dict):
            return None

        microns_per_px: Optional[float] = None
        if "MicronsPerPixel" in json_data:
            microns_per_px = _lenient_float(json_data["MicronsPerPixel"])
        elif "Target" in json_data and isinstance(json_data["Target"], dict):
            microns_per_px = _lenient_float(json_data["Target"].get("PixelSize"))
        elif "PixelSize" in json_data:
            microns_per_px = _lenient_float(json_data["PixelSize"])

        mag = None
        na = None
        obj_settings = json_data.get("ObjectiveSettings", {})
        if isinstance(obj_settings, str):
            try:
                obj_settings = json.loads(obj_settings)
            except json.JSONDecodeError:
                obj_settings = {}

        if isinstance(obj_settings, dict):
            mag = obj_settings.get("Magnification")
            na = obj_settings.get("NumericalAperture")

        if mag is None and "Magnification" in json_data:
            mag = json_data.get("Magnification")
        if na is None and "Objective NA" in json_data:
            na = json_data.get("Objective NA")

        software = json_data.get("Software", "EVOS")

        if microns_per_px and microns_per_px > 0:
            if _is_plausible(microns_per_px):
                return ScaleInfo(
                    pixel_size_um=microns_per_px,
                    source="evos_tag_37510",
                    magnification=_lenient_float(mag),
                    numerical_aperture=_lenient_float(na),
                    instrument_model=str(software),
                    is_calibrated=True,
                    extra_metadata=json_data,
                )
            return self._rejected(
                "evos_tag_37510 implausible",
                f"EVOS tag 37510 scale {microns_per_px} um/px outside plausible range "
                f"{list(PLAUSIBLE_UM_PER_PX)} um/px; using default {self.default_pixel_size_um} um/px",
                extra_metadata=json_data,
            )
        return None

    @staticmethod
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].split(":")[-1]

    def _find_local(self, root: ET.Element, name: str) -> Optional[ET.Element]:
        for el in root.iter():
            if isinstance(el.tag, str) and self._local(el.tag) == name:
                return el
        return None

    def _extract_ome_xml(self, tags: Dict[int, Any], image_size: Tuple[int, int]) -> Optional[ScaleInfo]:
        """Extract scale from OME-XML in Tag 270 (ImageDescription), any OME schema year / prefix."""
        if 270 not in tags:
            return None

        desc = _decode_text(tags[270])
        if not re.search(r"<(?:\w+:)?OME[\s>]", desc):
            return None

        try:
            # Remove comments anywhere (Bio-Formats headers start with one) and anything before
            # the root element (XML declaration, which may claim a different encoding).
            xml_str = re.sub(r"<!--.*?-->", "", desc, flags=re.DOTALL)
            start = re.search(r"<(?:\w+:)?OME[\s>]", xml_str)
            if start is None:
                return None
            xml_str = xml_str[start.start():].strip().rstrip("\x00")

            root = ET.fromstring(xml_str)
            pixels_elem = self._find_local(root, "Pixels")

            if pixels_elem is not None:
                phys_x = pixels_elem.attrib.get("PhysicalSizeX")
                if phys_x:
                    # OME PhysicalSizeX is the size of ONE pixel, in PhysicalSizeXUnit (default micrometre).
                    unit = pixels_elem.attrib.get("PhysicalSizeXUnit", "µm").strip()
                    factor = OME_LENGTH_TO_UM.get(unit)
                    if factor is None:
                        logger.warning(f"Unsupported OME PhysicalSizeXUnit {unit!r}; ignoring OME scale")
                        return None
                    pixel_size = float(phys_x) * factor

                    microscope_elem = self._find_local(root, "Microscope")
                    model = microscope_elem.attrib.get("Model") if microscope_elem is not None else "OME Microscope"

                    if _is_plausible(pixel_size):
                        return ScaleInfo(
                            pixel_size_um=pixel_size,
                            source="OME-XML Tag 270",
                            instrument_model=model,
                            is_calibrated=True,
                        )
                    return self._rejected(
                        "OME-XML Tag 270 implausible",
                        f"OME pixel size {pixel_size} um/px outside plausible range {PLAUSIBLE_UM_PER_PX}; "
                        f"using default {self.default_pixel_size_um} um/px",
                        instrument_model=model,
                    )
        except Exception as e:
            logger.debug(f"Failed parsing OME-XML: {e}")

        return None

    @staticmethod
    def _rational(val: Any) -> Optional[float]:
        if isinstance(val, tuple) and len(val) == 2:
            try:
                return float(val[0]) / float(val[1]) if float(val[1]) != 0 else None
            except (TypeError, ValueError):
                return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)  # PIL IFDRational
        except (TypeError, ValueError):
            return None

    def _extract_imagej(self, tags: Dict[int, Any], info_dict: Dict[str, Any]) -> Optional[ScaleInfo]:
        """Extract XY calibration from ImageJ / Fiji metadata.

        ImageJ stores the XY scale either as ``pixelWidth=`` in the description or (the usual
        case for files saved by ImageJ/Fiji) as XResolution (pixels per ``unit``) with TIFF
        ResolutionUnit NONE. The ImageJ ``unit`` is converted to micrometres.
        """
        desc = _decode_text(tags.get(270, info_dict.get("description", "")))
        is_imagej = desc.lstrip().startswith("ImageJ=") or "\nImageJ=" in desc
        if not (is_imagej or "unit=" in desc):
            return None

        unit_match = re.search(r"(?m)^\s*unit=(.*?)\s*$", desc) or re.search(r"unit=([^\s,;]+)", desc)
        unit = unit_match.group(1) if unit_match else None

        # 'spacing' is the Z-slice spacing, NOT the XY pixel size. Only pixelWidth / XResolution are XY.
        xy_match = re.search(r"pixelWidth=([0-9.eE+-]+)", desc) or re.search(r"dxf=([0-9.eE+-]+)", desc)
        scale_in_unit: Optional[float] = None
        origin = "pixelWidth"
        if xy_match:
            scale_in_unit = _lenient_float(xy_match.group(1))
        else:
            res_unit = tags.get(296, info_dict.get("resolution_unit", 1))
            try:
                res_unit = int(res_unit)
            except (TypeError, ValueError):
                res_unit = 1
            res = self._rational(tags.get(282)) if 282 in tags else None
            # Only interpret XResolution as pixels-per-ImageJ-unit when the TIFF unit is NONE;
            # inch/cm resolution tags are handled by the standard TIFF strategy.
            if res_unit == 1 and res and res > 0 and unit is not None:
                scale_in_unit = 1.0 / res
                origin = "XResolution"

        if scale_in_unit is None or not scale_in_unit > 0:
            return None

        if unit is None:
            unit = "um" if origin == "pixelWidth" else ""
        known, factor = _imagej_unit_factor(unit)
        if not known:
            return self._rejected(
                f"ImageJ unknown unit {unit!r}",
                f"ImageJ calibration unit {unit!r} not recognised; using default {self.default_pixel_size_um} um/px",
            )
        if factor is None:
            return self._rejected(
                "ImageJ unit=pixel",
                f"ImageJ calibration unit is {unit!r} (uncalibrated); using default {self.default_pixel_size_um} um/px",
            )

        px_size_um = scale_in_unit * factor
        if _is_plausible(px_size_um):
            return ScaleInfo(pixel_size_um=px_size_um, source="ImageJ metadata", is_calibrated=True)
        return self._rejected(
            "ImageJ metadata implausible",
            f"ImageJ XY scale {px_size_um:.4g} um/px falls outside plausible range {list(PLAUSIBLE_UM_PER_PX)}; "
            f"using default {self.default_pixel_size_um} um/px",
        )

    def _extract_tiff_resolution(self, tags: Dict[int, Any], info_dict: Dict[str, Any]) -> Optional[ScaleInfo]:
        """Extract calibration from standard TIFF resolution tags."""
        unit_val = tags.get(296, info_dict.get("resolution_unit", 1))
        try:
            unit_val = int(unit_val)
        except (TypeError, ValueError):
            unit_val = 1
        res_x = tags.get(282, info_dict.get("dpi", (1, 1))[0] if isinstance(info_dict.get("dpi"), tuple) else None)

        if res_x is None:
            return None
        res_val = self._rational(res_x)
        if not res_val or res_val <= 1.0:
            return None

        # Disallow 72 / 96 DPI desktop default resolution tags
        if unit_val == 2 and any(abs(res_val - d) < 1.0 for d in (72.0, 96.0)):
            return self._rejected(
                f"{round(res_val)} DPI screen default",
                f"TIFF Resolution tag reports {round(res_val)} DPI (uncalibrated screen default). "
                f"Rejecting calibration; using default {self.default_pixel_size_um} um/px",
            )

        # unit_val: 2 = inches, 3 = cm
        if unit_val == 3:  # pixels / cm
            px_size_um = 10000.0 / res_val
        elif unit_val == 2:  # pixels / inch
            px_size_um = 25400.0 / res_val
        else:
            return None

        if not _is_plausible(px_size_um):
            return self._rejected(
                "TIFF Resolution Tags implausible",
                f"TIFF Resolution scale {px_size_um:.4g} um/px falls outside plausible range "
                f"{list(PLAUSIBLE_UM_PER_PX)}; using default {self.default_pixel_size_um} um/px",
            )

        return ScaleInfo(
            pixel_size_um=px_size_um,
            source="TIFF Resolution Tags",
            is_calibrated=True,
        )


def extract_scale_info(
    image_path: str | Path,
    override_um: Optional[float] = None,
    default_um: float = 1.518817,
) -> ScaleInfo:
    """Full :class:`ScaleInfo` (pixel size, source, ``is_calibrated``, warning) for an image."""
    return ScaleExtractor(default_pixel_size_um=default_um, cli_override_um=override_um).extract(image_path)


@overload
def extract_pixel_size(
    image_path: str | Path,
    override_um: Optional[float] = ...,
    default_um: float = ...,
    return_info: Literal[False] = ...,
) -> Tuple[float, str]: ...


@overload
def extract_pixel_size(
    image_path: str | Path,
    override_um: Optional[float] = ...,
    default_um: float = ...,
    *,
    return_info: Literal[True],
) -> ScaleInfo: ...


def extract_pixel_size(
    image_path: str | Path,
    override_um: Optional[float] = None,
    default_um: float = 1.518817,
    return_info: bool = False,
) -> Union[Tuple[float, str], ScaleInfo]:
    """
    Interface contract function for optical pixel size extraction.

    Returns ``(pixel_size_um, source_description)``; with ``return_info=True`` returns the
    full :class:`ScaleInfo` (including ``is_calibrated``) instead.
    """
    info = extract_scale_info(image_path, override_um=override_um, default_um=default_um)
    if return_info:
        return info
    return info.pixel_size_um, info.source
