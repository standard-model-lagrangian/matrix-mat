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
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import tifffile

logger = logging.getLogger("spheroid_pipeline_v2.scale_extractor")


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

    def extract(self, image_path: str | Path) -> ScaleInfo:
        """
        Extract pixel size (um/pixel) and metadata from an image file.

        Priority order:
        1. CLI override if specified (--pixel-size)
        2. Embedded EVOS JSON (TIFF Tag 37510 or Tag 270 comment)
        3. OME-XML (Tag 270 PhysicalSizeX / PhysicalSizeY)
        4. ImageJ metadata tags
        5. Standard TIFF resolution tags (Tags 282, 283, 296)
        6. Default config fallback (with warning log)
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

        # Attempt extraction via tifffile first (most accurate for scientific TIFFs)
        try:
            with tifffile.TiffFile(image_path) as tif:
                for page in tif.pages:
                    tags = {tag.code: tag.value for tag in page.tags.values()}

                    # Strategy 2: Check EVOS Tag 37510
                    evos_info = self._extract_evos_metadata(tags)
                    if evos_info is not None:
                        return evos_info

                    # Strategy 3: Check OME-XML (Tag 270)
                    ome_info = self._extract_ome_xml(tags, (page.shape[1], page.shape[0]) if len(page.shape) >= 2 else (2048, 1536))
                    if ome_info is not None:
                        return ome_info

                    # Strategy 4: Check ImageJ metadata
                    imagej_info = self._extract_imagej(tags, {})
                    if imagej_info is not None:
                        return imagej_info

                    # Strategy 5: Check standard TIFF resolution tags
                    tiff_res_info = self._extract_tiff_resolution(tags, {})
                    if tiff_res_info is not None:
                        return tiff_res_info
        except Exception as e:
            logger.debug(f"tifffile extraction encounter: {e}, attempting PIL fallback...")

        # Fallback to PIL inspection if tifffile did not resolve
        try:
            with Image.open(image_path) as im:
                tags = getattr(im, "tag_v2", {})
                info_dict = getattr(im, "info", {})

                evos_info = self._extract_evos_metadata(tags)
                if evos_info is not None:
                    return evos_info

                ome_info = self._extract_ome_xml(tags, im.size)
                if ome_info is not None:
                    return ome_info

                imagej_info = self._extract_imagej(tags, info_dict)
                if imagej_info is not None:
                    return imagej_info

                tiff_res_info = self._extract_tiff_resolution(tags, info_dict)
                if tiff_res_info is not None:
                    return tiff_res_info
        except Exception as e:
            logger.warning(f"Error reading metadata from {image_path.name}: {e}")

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
            val = tags[37510]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            elif not isinstance(val, str):
                val = str(val)

            m = re.search(r"(\{.*\})", val, re.DOTALL)
            if m:
                try:
                    json_data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Check Tag 270 comment block for EVOS JSON
        if json_data is None and 270 in tags:
            desc = tags[270]
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
            elif not isinstance(desc, str):
                desc = str(desc)

            m = re.search(r"<!--\s*(\{.*?\})\s*-->", desc, re.DOTALL)
            if m:
                try:
                    json_data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if json_data:
            microns_per_px = None
            if "MicronsPerPixel" in json_data:
                try:
                    microns_per_px = float(json_data["MicronsPerPixel"])
                except (ValueError, TypeError):
                    pass

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
                return ScaleInfo(
                    pixel_size_um=microns_per_px,
                    source="TIFF Tag 37510 JSON",
                    magnification=float(mag) if mag is not None else None,
                    numerical_aperture=float(na) if na is not None else None,
                    instrument_model=str(software),
                    is_calibrated=True,
                    extra_metadata=json_data,
                )
        return None

    def _extract_ome_xml(self, tags: Dict[int, Any], image_size: Tuple[int, int]) -> Optional[ScaleInfo]:
        """Extract scale from OME-XML in Tag 270 (ImageDescription)."""
        if 270 not in tags:
            return None

        desc = tags[270]
        if isinstance(desc, bytes):
            desc = desc.decode("utf-8", errors="ignore")
        elif not isinstance(desc, str):
            desc = str(desc)

        if "<OME" not in desc:
            return None

        try:
            xml_str = desc
            if "<!--" in xml_str:
                xml_str = xml_str.split("<!--")[0].strip()

            root = ET.fromstring(xml_str)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels_elem = root.find(".//ome:Pixels", ns)
            if pixels_elem is None:
                pixels_elem = root.find(".//Pixels")

            if pixels_elem is not None:
                phys_x = pixels_elem.attrib.get("PhysicalSizeX")
                size_x = pixels_elem.attrib.get("SizeX", str(image_size[0]))

                if phys_x:
                    px_val = float(phys_x)
                    if px_val > 100 and float(size_x) > 100:
                        pixel_size = px_val / float(size_x)
                    else:
                        pixel_size = px_val

                    microscope_elem = root.find(".//ome:Microscope", ns) or root.find(".//Microscope")
                    model = microscope_elem.attrib.get("Model") if microscope_elem is not None else "OME Microscope"

                    return ScaleInfo(
                        pixel_size_um=pixel_size,
                        source="OME-XML Tag 270",
                        instrument_model=model,
                        is_calibrated=True,
                    )
        except Exception as e:
            logger.debug(f"Failed parsing OME-XML: {e}")

        return None

    def _extract_imagej(self, tags: Dict[int, Any], info_dict: Dict[str, Any]) -> Optional[ScaleInfo]:
        """Extract calibration from ImageJ metadata."""
        raw_desc = tags.get(270, info_dict.get("description", ""))
        if isinstance(raw_desc, bytes):
            raw_desc = raw_desc.decode("utf-8", errors="ignore")
        desc = str(raw_desc)

        if "ImageJ" in desc or "unit=" in desc:
            unit_match = re.search(r"unit=([a-zA-Zµu]+)", desc)
            unit = unit_match.group(1) if unit_match else "um"

            spacing_match = re.search(r"spacing=([0-9.]+)", desc) or re.search(r"pixelWidth=([0-9.]+)", desc)
            if spacing_match:
                scale = float(spacing_match.group(1))
                if unit in ["um", "µm", "micron", "microns"]:
                    return ScaleInfo(pixel_size_um=scale, source="ImageJ metadata", is_calibrated=True)
                elif unit in ["mm"]:
                    return ScaleInfo(pixel_size_um=scale * 1000.0, source="ImageJ metadata", is_calibrated=True)
        return None

    def _extract_tiff_resolution(self, tags: Dict[int, Any], info_dict: Dict[str, Any]) -> Optional[ScaleInfo]:
        """Extract calibration from standard TIFF resolution tags."""
        unit_val = tags.get(296, info_dict.get("resolution_unit", 1))
        res_x = tags.get(282, info_dict.get("dpi", (1, 1))[0] if isinstance(info_dict.get("dpi"), tuple) else None)

        if res_x is not None:
            if isinstance(res_x, tuple) and len(res_x) == 2 and res_x[1] != 0:
                res_val = res_x[0] / res_x[1]
            elif isinstance(res_x, (int, float)):
                res_val = float(res_x)
            else:
                res_val = None

            if res_val and res_val > 1.0:
                # unit_val: 2 = inches, 3 = cm
                if unit_val == 3:  # pixels / cm
                    px_size_um = 10000.0 / res_val
                elif unit_val == 2:  # pixels / inch
                    px_size_um = 25400.0 / res_val
                else:
                    return None

                return ScaleInfo(
                    pixel_size_um=px_size_um,
                    source="TIFF Resolution Tags",
                    is_calibrated=True,
                )
        return None


def extract_pixel_size(
    image_path: str | Path,
    override_um: Optional[float] = None,
    default_um: float = 1.518817,
) -> Tuple[float, str]:
    """
    Interface contract function for optical pixel size extraction.
    Returns (pixel_size_um, source_description).
    """
    extractor = ScaleExtractor(default_pixel_size_um=default_um, cli_override_um=override_um)
    info = extractor.extract(image_path)
    return info.pixel_size_um, info.source
