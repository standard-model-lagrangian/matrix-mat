"""
File discovery, filename metadata parsing, SHA-256 calculation, and channel pairing.

Discovery is shared with preflight (`discover_channel_files`) so both see exactly the same files:
case-insensitive .tif/.tiff, channel taken from an exact `_chNN` suffix.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import tifffile
from matrix_mat.core.scale import ScaleExtractor

logger = logging.getLogger("matrix_mat.if_density.pairing")

NUCLEAR_CHANNEL = "00"
ACTIN_CHANNEL = "02"

# '<base>_chNN.tif' / '.tiff' (any case). The channel must be the exact suffix.
CHANNEL_FILE_PATTERN = re.compile(r"^(?P<base>.+)_ch(?P<channel>\d{2})\.tiff?$", re.IGNORECASE)

# '<cell line> Spheroid D<day> [<tag>]_<material>-Gel<n>[-<subfield>]', e.g.
#   'SKOV3 Spheroid D7 Mor_S34D30-Gel8-1-2', 'OVCAR3 Spheroid D14_S4.5D30-Gel2', '... Mor_S40_D30-Gel1-1'.
# Material may contain letters, digits, dots and underscores; it ends at the first '-Gel<n>'.
FILENAME_PATTERN = re.compile(
    r"^(?P<cell_line>\S+)\s+Spheroid\s+D(?P<day>\d+(?:\.\d+)?)(?:\s+(?P<tag>[^_]+?))?\s*_"
    r"(?P<material>[A-Za-z0-9][A-Za-z0-9._]*?)-(?P<replicate>[Gg]el\d+)(?:-(?P<subfield>.+))?$"
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
    segmentation_mode: str  # always '2D' (3D inputs are rejected)
    cell_line: str = ""
    day: str = ""
    scale_source: str = ""
    is_calibrated: bool = True


@dataclass
class ChannelDiscovery:
    """Channel files found in a data directory, keyed by field base name."""
    files: List[Path]
    channels: Dict[str, Dict[str, Path]]  # channel ('00') -> base -> path
    unrecognized: List[Path] = field(default_factory=list)
    collisions: List[str] = field(default_factory=list)

    @property
    def nuclear(self) -> Dict[str, Path]:
        return self.channels.get(NUCLEAR_CHANNEL, {})

    @property
    def actin(self) -> Dict[str, Path]:
        return self.channels.get(ACTIN_CHANNEL, {})


def discover_channel_files(data_dir: str | Path) -> ChannelDiscovery:
    """
    List .tif/.tiff files (case-insensitive) and index them by channel and base name.
    Two files that resolve to the same base and channel (e.g. X_ch00.tif and X_ch00.tiff) are a collision.
    """
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    channels: Dict[str, Dict[str, Path]] = {}
    unrecognized: List[Path] = []
    collisions: List[str] = []
    for p in files:
        m = CHANNEL_FILE_PATTERN.match(p.name)
        if not m:
            unrecognized.append(p)
            continue
        base, ch = m.group("base"), m.group("channel")
        slot = channels.setdefault(ch, {})
        if base in slot:
            collisions.append(f"{base} ch{ch}: {slot[base].name} and {p.name}")
            continue
        slot[base] = p
    return ChannelDiscovery(files=files, channels=channels, unrecognized=unrecognized, collisions=collisions)


def parse_field_name(base_name: str) -> Dict[str, str]:
    """
    Parse cell line, day, material, replicate (gel) and subfield from a field base name.
    Raises ValueError when the name does not follow the expected pattern.
    """
    match = FILENAME_PATTERN.match(base_name)
    if not match:
        raise ValueError(
            f"Unrecognised IF field name '{base_name}': expected "
            "'<cell line> Spheroid D<day> [<tag>]_<material>-Gel<n>[-<subfield>]'"
        )
    return {
        "cell_line": match.group("cell_line"),
        "day": match.group("day"),
        "material": match.group("material"),
        "replicate": match.group("replicate"),
        "subfield": match.group("subfield") or "1",
    }


def parse_field_metadata(base_name: str) -> Tuple[str, str, str]:
    """
    Parse (material, replicate, subfield) from base name; raises ValueError on an unrecognised name.
    Example: 'SKOV3 Spheroid D7 Mor_Mat-Gel1-2' -> ('Mat', 'Gel1', '2')
    """
    info = parse_field_name(base_name)
    return info["material"], info["replicate"], info["subfield"]


def read_image_2d(path: Path, image_id: str = ""):
    """Read a TIFF as a 2D array: RGB(A) is converted to grayscale explicitly, z-stacks are refused."""
    from matrix_mat.if_density.preprocess import ensure_2d_grayscale
    return ensure_2d_grayscale(tifffile.imread(path), image_id or path.name)


def classify_shape(shape: Tuple[int, ...]) -> str:
    """'2D', 'RGB' (H, W, 3|4), or '3D' (anything else with >2 dims and more than one plane)."""
    if len(shape) == 2:
        return "2D"
    if len(shape) == 3 and shape[-1] in (3, 4) and shape[0] > 4:
        return "RGB"
    if len(shape) >= 3 and all(s == 1 for s in shape[:-2]):
        return "2D"
    return "3D"


def spatial_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """(H, W) of a 2D, singleton-stack or RGB(A) image shape."""
    return tuple(shape[:2]) if classify_shape(shape) == "RGB" else tuple(shape[-2:])


@dataclass
class PairingResult:
    records: List[FieldRecord]
    unpaired_bases: List[str]
    unparsed_bases: List[str]


def discover_and_pair_fields_detailed(
    data_dir: str | Path,
    allow_unpaired: bool = False,
    default_pixel_size_um: float = 0.505049,
    allow_uncalibrated: bool = True,
) -> PairingResult:
    """
    Discover TIFF files in data_dir, pair ch00 (nuclear) and ch02 (actin), compute hashes and metadata.

    - Fields whose names do not parse are skipped with a warning (never pooled into an 'Unknown' group).
    - When the data contain more than one (cell line, day) group, the material label is qualified as
      '<cell line>_D<day>_<material>' so different groups are not pooled.
    - A field whose nuclear TIFF has no physical scale raises unless allow_uncalibrated is True;
      the scale source is recorded per field.
    - z-stacks are rejected (the pipeline is 2D only).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    disc = discover_channel_files(data_dir)
    if not disc.files:
        raise ValueError(f"No TIFF files found in {data_dir}")
    if disc.collisions:
        raise ValueError("Ambiguous channel files (same field and channel twice):\n  " + "\n  ".join(disc.collisions))
    for p in disc.unrecognized:
        logger.warning(f"Unrecognized TIFF name (no _chNN suffix): {p.name}")
    for ch in sorted(set(disc.channels) - {NUCLEAR_CHANNEL, ACTIN_CHANNEL}):
        logger.warning(f"Ignoring {len(disc.channels[ch])} file(s) of channel ch{ch} (only ch00 and ch02 are used)")

    ch00_files, ch02_files = disc.nuclear, disc.actin
    all_bases = sorted(set(ch00_files) | set(ch02_files))
    scale_extractor = ScaleExtractor(default_pixel_size_um=default_pixel_size_um)

    paired_records: List[FieldRecord] = []
    unpaired_bases: List[str] = []
    unparsed_bases: List[str] = []
    uncalibrated: List[str] = []

    for base in all_bases:
        if not (base in ch00_files and base in ch02_files):
            missing = "ch00" if base not in ch00_files else "ch02"
            logger.warning(f"Field '{base}' is unpaired: missing {missing} channel")
            unpaired_bases.append(base)
            continue
        try:
            info = parse_field_name(base)
        except ValueError as e:
            logger.warning(f"Skipping field: {e}")
            unparsed_bases.append(base)
            continue

        nuc_path = ch00_files[base]
        act_path = ch02_files[base]

        with tifffile.TiffFile(nuc_path) as tif:
            shape = tuple(tif.series[0].shape)
        with tifffile.TiffFile(act_path) as tif:
            act_shape = tuple(tif.series[0].shape)
        kind = classify_shape(shape)
        if kind == "3D" or classify_shape(act_shape) == "3D":
            raise ValueError(
                f"Field '{base}' is a multi-plane stack {shape}; 3D input is not supported by the IF pipeline"
            )
        if spatial_shape(shape) != spatial_shape(act_shape):
            raise ValueError(f"Field '{base}': ch00 shape {shape} does not match ch02 shape {act_shape}")

        scale_info = scale_extractor.extract(nuc_path)
        if not scale_info.is_calibrated:
            uncalibrated.append(base)
        act_scale = scale_extractor.extract(act_path)
        if (scale_info.is_calibrated and act_scale.is_calibrated
                and abs(act_scale.pixel_size_um - scale_info.pixel_size_um) > 0.01 * scale_info.pixel_size_um):
            raise ValueError(
                f"Field '{base}': ch00 pixel size {scale_info.pixel_size_um} != ch02 pixel size {act_scale.pixel_size_um}"
            )

        paired_records.append(FieldRecord(
            field_id=base,
            material=info["material"],
            replicate=info["replicate"],
            subfield=info["subfield"],
            nuclear_path=nuc_path,
            actin_path=act_path,
            nuclear_sha256=compute_sha256(nuc_path),
            actin_sha256=compute_sha256(act_path),
            image_shape=shape,
            pixel_size_um=scale_info.pixel_size_um,
            segmentation_mode="2D",
            cell_line=info["cell_line"],
            day=info["day"],
            scale_source=scale_info.source,
            is_calibrated=bool(scale_info.is_calibrated),
        ))

    if uncalibrated:
        msg = (
            f"{len(uncalibrated)} field(s) have no embedded pixel scale and would use the default "
            f"{default_pixel_size_um} um/px: " + ", ".join(uncalibrated)
        )
        if not allow_uncalibrated:
            raise ValueError(msg + ". Set allow_uncalibrated: true in the config to accept the default scale.")
        logger.warning(msg + " (allow_uncalibrated=true)")

    groups = sorted({(r.cell_line, r.day) for r in paired_records})
    if len(groups) > 1:
        logger.warning(
            f"Data contain {len(groups)} (cell line, day) groups {groups}; material labels are qualified as "
            "'<cell line>_D<day>_<material>' so groups are not pooled"
        )
        for r in paired_records:
            r.material = f"{r.cell_line}_D{r.day}_{r.material}"

    if unpaired_bases and not allow_unpaired:
        msg = (
            f"Found {len(unpaired_bases)} unpaired fields in {data_dir}:\n"
            + "\n".join(f"  - {b}" for b in unpaired_bases)
            + "\nSet allow_unpaired=True or pass --allow-unpaired to ignore and continue."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        f"Pairing complete: {len(paired_records)} valid pairs, {len(unpaired_bases)} unpaired skipped, "
        f"{len(unparsed_bases)} unrecognised names skipped."
    )
    return PairingResult(records=paired_records, unpaired_bases=unpaired_bases, unparsed_bases=unparsed_bases)


def discover_and_pair_fields(
    data_dir: str | Path,
    allow_unpaired: bool = False,
    default_pixel_size_um: float = 0.505049,
    allow_uncalibrated: bool = True,
) -> Tuple[List[FieldRecord], List[str]]:
    """Backwards-compatible wrapper: returns (records, unpaired_bases)."""
    res = discover_and_pair_fields_detailed(
        data_dir, allow_unpaired=allow_unpaired, default_pixel_size_um=default_pixel_size_um,
        allow_uncalibrated=allow_uncalibrated,
    )
    return res.records, res.unpaired_bases
