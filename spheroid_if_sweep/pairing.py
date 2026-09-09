"""
File discovery, regex metadata parsing, SHA-256 calculation, and channel pairing.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tifffile
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor

logger = logging.getLogger("spheroid_if_sweep.pairing")

FILENAME_PATTERN = re.compile(
    r"^SKOV3 Spheroid D7 Mor_(?P<material>[A-Za-z0-9]+)-(?P<sample>Gel\d+.*)$"
)


def compute_sha256(file_path: Path) -> str:
    """Compute sha256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class FieldRecord:
    field_id: str
    material: str
    replicate: str
    subfield: str
    nuclear_path: Path
    actin_path: Path
    nuclear_sha256: str
    actin_sha256: str
    image_shape: Tuple[int, ...]
    pixel_size_um: float
    segmentation_mode: str  # '2D' or '3D'


def parse_field_metadata(base_name: str) -> Tuple[str, str, str]:
    """
    Parse material, replicate, and subfield from base name.
    Example: 'SKOV3 Spheroid D7 Mor_Mat-Gel1-2' -> ('Mat', 'Gel1', '2')
    """
    match = FILENAME_PATTERN.match(base_name)
    if not match:
        return "Unknown", "Unknown", "1"

    material = match.group("material")
    sample = match.group("sample")
    parts = sample.split("-")
    replicate = parts[0]
    subfield = "-".join(parts[1:]) if len(parts) > 1 else "1"
    return material, replicate, subfield


def discover_and_pair_fields(
    data_dir: str | Path,
    allow_unpaired: bool = False,
    default_pixel_size_um: float = 0.505049,
) -> Tuple[List[FieldRecord], List[str]]:
    """
    Discover all TIFF files in data_dir, pair ch00 (nuclear) and ch02 (actin),
    compute hashes and metadata, and validate pairing completeness.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_tifs = sorted(list(data_dir.glob("*.tif")))
    if not all_tifs:
        raise ValueError(f"No TIFF files found in {data_dir}")

    ch00_files: Dict[str, Path] = {}
    ch02_files: Dict[str, Path] = {}

    for p in all_tifs:
        if p.name.endswith("_ch00.tif"):
            base = p.name[:-9]
            ch00_files[base] = p
        elif p.name.endswith("_ch02.tif"):
            base = p.name[:-9]
            ch02_files[base] = p
        else:
            logger.warning(f"Unrecognized TIFF suffix (not ch00 or ch02): {p.name}")

    all_bases = sorted(set(list(ch00_files.keys()) + list(ch02_files.keys())))
    scale_extractor = ScaleExtractor(default_pixel_size_um=default_pixel_size_um)

    paired_records: List[FieldRecord] = []
    unpaired_bases: List[str] = []

    for base in all_bases:
        has_ch00 = base in ch00_files
        has_ch02 = base in ch02_files

        if not (has_ch00 and has_ch02):
            missing = "ch00" if not has_ch00 else "ch02"
            logger.warning(f"Field '{base}' is unpaired: missing {missing} channel")
            unpaired_bases.append(base)
            continue

        nuc_path = ch00_files[base]
        act_path = ch02_files[base]

        material, replicate, subfield = parse_field_metadata(base)

        # Extract image shape from nuclear file
        with tifffile.TiffFile(nuc_path) as tif:
            shape = tuple(tif.series[0].shape)

        seg_mode = "3D" if len(shape) >= 3 and shape[0] > 1 else "2D"

        # Extract pixel size
        scale_info = scale_extractor.extract(nuc_path)
        pixel_size = scale_info.pixel_size_um

        # Calculate sha256 checksums
        nuc_sha = compute_sha256(nuc_path)
        act_sha = compute_sha256(act_path)

        rec = FieldRecord(
            field_id=base,
            material=material,
            replicate=replicate,
            subfield=subfield,
            nuclear_path=nuc_path,
            actin_path=act_path,
            nuclear_sha256=nuc_sha,
            actin_sha256=act_sha,
            image_shape=shape,
            pixel_size_um=pixel_size,
            segmentation_mode=seg_mode,
        )
        paired_records.append(rec)

    if unpaired_bases and not allow_unpaired:
        msg = (
            f"Found {len(unpaired_bases)} unpaired fields in {data_dir}:\n"
            + "\n".join(f"  - {b}" for b in unpaired_bases)
            + "\nSet allow_unpaired=True or pass --allow-unpaired to ignore and continue."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        f"Pairing complete: {len(paired_records)} valid pairs, {len(unpaired_bases)} unpaired skipped."
    )
    return paired_records, unpaired_bases
