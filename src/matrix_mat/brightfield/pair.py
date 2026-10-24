"""
Temporal Pairing & Trajectory Tracking Module for Brightfield Spheroids.

Provides:
  - Mutual nearest centroid bipartite matching across timepoints (Day 0 -> Day 7)
    using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
  - Maximum displacement tolerance gating:
      D_max = max(min_dist_px, max_shift_fraction * mean_radius_px)
  - Denominator gating (d_t0 >= 60.0 um) to prevent near-zero baseline fold change distortion.
  - Trajectory fold-change gating (V7 / V0 in [0.1, 20.0]) and composite QC flagging.
  - SpheroidPairRecord dataclass matching the full 30-column pairs.csv schema.
  - Helper functions for dataset-level pairing, updating object records with pair IDs,
    and persisting pairs.csv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from matrix_mat.brightfield.config import PairingConfig, PipelineConfig
from matrix_mat.brightfield.measure import SpheroidObjectRecord
from matrix_mat.core.naming import parse_timepoint

logger = logging.getLogger("matrix_mat.brightfield.pair")

# Exact 30-column header sequence mandated by schema
PAIRS_CSV_COLUMNS: List[str] = [
    "pair_id",
    "pair_key",
    "condition",
    "replicate",
    "fov",
    "t0_object_id",
    "t0_centroid_x",
    "t0_centroid_y",
    "t0_eq_diameter_um",
    "t0_volume_sphere_um3",
    "t0_volume_ellipsoid_um3",
    "t0_circularity",
    "t0_qc_flag",
    "t7_object_id",
    "t7_centroid_x",
    "t7_centroid_y",
    "t7_eq_diameter_um",
    "t7_volume_sphere_um3",
    "t7_volume_ellipsoid_um3",
    "t7_circularity",
    "t7_qc_flag",
    "centroid_distance_px",
    "delta_volume_um3",
    "fold_change_volume",
    "log2_fold_change",
    "delta_diameter_um",
    "fold_change_diameter",
    "denominator_gate_pass",
    "combined_qc_flag",
    "qc_reasons",
]


@dataclass
class SpheroidPairRecord:
    """Dataclass representing a tracked spheroid pair between Day 0 and Day 7 matching pairs.csv schema."""
    pair_id: str
    pair_key: str
    condition: str
    replicate: str
    fov: str
    # t0 metrics
    t0_object_id: str
    t0_centroid_x: float
    t0_centroid_y: float
    t0_eq_diameter_um: float
    t0_volume_sphere_um3: float
    t0_volume_ellipsoid_um3: float
    t0_circularity: float
    t0_qc_flag: str
    # t7 metrics
    t7_object_id: str
    t7_centroid_x: float
    t7_centroid_y: float
    t7_eq_diameter_um: float
    t7_volume_sphere_um3: float
    t7_volume_ellipsoid_um3: float
    t7_circularity: float
    t7_qc_flag: str
    # Pair trajectory metrics
    centroid_distance_px: float
    delta_volume_um3: float
    fold_change_volume: float
    log2_fold_change: float
    delta_diameter_um: float
    fold_change_diameter: float
    denominator_gate_pass: bool
    combined_qc_flag: str  # PASS / REVIEW / FAIL
    qc_reasons: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to ordered dictionary matching PAIRS_CSV_COLUMNS."""
        raw = asdict(self)
        return {col: raw[col] for col in PAIRS_CSV_COLUMNS}


class SpheroidPairer:
    """
    Hungarian bipartite matching engine for tracking spheroids across timepoints.
    """

    def __init__(self, config: Optional[Union[PairingConfig, PipelineConfig]] = None):
        if isinstance(config, PipelineConfig):
            self.config = config.pairing
        elif isinstance(config, PairingConfig):
            self.config = config
        else:
            self.config = PairingConfig()

    def pair_fov_objects(
        self,
        t0_objects: List[SpheroidObjectRecord],
        t7_objects: List[SpheroidObjectRecord],
        pair_key: Optional[str] = None,
    ) -> Tuple[List[SpheroidPairRecord], List[SpheroidObjectRecord], List[SpheroidObjectRecord]]:
        """
        Pairs t0 and t7 objects within a single FOV/well using Hungarian linear sum assignment.

        Args:
            t0_objects: List of segmented objects from Day 0
            t7_objects: List of segmented objects from Day 7
            pair_key: Optional identifier for well/FOV (e.g. 'Mat_gel1_0001')

        Returns:
            Tuple of (paired_records, unmatched_t0_objects, unmatched_t7_objects)
        """
        if not t0_objects or not t7_objects:
            return [], list(t0_objects), list(t7_objects)

        derived_pair_key = pair_key or t0_objects[0].pair_key

        n0 = len(t0_objects)
        n7 = len(t7_objects)

        # Construct Euclidean centroid distance cost matrix
        c0 = np.array([[obj.centroid_x, obj.centroid_y] for obj in t0_objects], dtype=np.float64)
        c7 = np.array([[obj.centroid_x, obj.centroid_y] for obj in t7_objects], dtype=np.float64)
        dist_matrix = np.linalg.norm(c0[:, None, :] - c7[None, :, :], axis=2)

        # Solve optimal linear sum assignment with greedy fallback
        try:
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
        except Exception as e:
            if getattr(self.config, "fallback_greedy", True):
                logger.warning(f"Hungarian assignment failed ({e}); using greedy fallback matching")
                r_list, c_list = [], []
                used_cols = set()
                for r_idx in range(n0):
                    best_c = None
                    best_val = float("inf")
                    for c_idx in range(n7):
                        if c_idx not in used_cols and dist_matrix[r_idx, c_idx] < best_val:
                            best_val = dist_matrix[r_idx, c_idx]
                            best_c = c_idx
                    if best_c is not None:
                        r_list.append(r_idx)
                        c_list.append(best_c)
                        used_cols.add(best_c)
                row_ind = np.array(r_list, dtype=int)
                col_ind = np.array(c_list, dtype=int)
            else:
                raise

        paired_records: List[SpheroidPairRecord] = []
        matched_t0_indices = set()
        matched_t7_indices = set()

        for r, c in zip(row_ind, col_ind):
            obj0 = t0_objects[r]
            obj7 = t7_objects[c]
            d_px = float(dist_matrix[r, c])

            # Calculate radius-scaled maximum displacement limit
            mean_r_px = (float(obj0.equivalent_diameter_px) + float(obj7.equivalent_diameter_px)) / 4.0
            max_dist_px = max(
                float(self.config.max_displacement_px_floor),
                mean_r_px * float(self.config.max_centroid_shift_fraction),
            )

            if d_px <= max_dist_px:
                matched_t0_indices.add(r)
                matched_t7_indices.add(c)

                # Volumetric trajectory metrics
                v0 = float(obj0.volume_sphere_um3)
                v7 = float(obj7.volume_sphere_um3)
                delta_v = v7 - v0
                fc_v = v7 / max(v0, 1e-6)
                if v7 <= 0.0 or fc_v <= 0.0:
                    log2_fc = float("nan")
                else:
                    log2_fc = float(math.log2(fc_v))

                # Diameter trajectory metrics
                d0 = float(obj0.equivalent_diameter_um)
                d7 = float(obj7.equivalent_diameter_um)
                delta_d = d7 - d0
                fc_d = d7 / max(d0, 1e-6)

                # Denominator Gate (d0 >= 60.0 um)
                denom_pass = bool(d0 >= float(self.config.min_t0_d_um_denominator))

                # Combined QC Flag evaluation
                reasons: List[str] = []
                if v7 <= 0.0 or fc_v <= 0.0:
                    reasons.append("VANISHED_SPHEROID: v7 <= 0")
                if obj0.qc_flag in ("FAIL", "SEG_FAIL"):
                    reasons.append(f"t0_{obj0.qc_flag}: {obj0.qc_reasons}")
                if obj7.qc_flag in ("FAIL", "SEG_FAIL"):
                    reasons.append(f"t7_{obj7.qc_flag}: {obj7.qc_reasons}")

                if obj0.qc_flag in ("FAIL", "SEG_FAIL") or obj7.qc_flag in ("FAIL", "SEG_FAIL") or v7 <= 0.0:
                    comb_qc = "FAIL"
                else:
                    if obj0.qc_flag == "REVIEW":
                        reasons.append(f"t0_REVIEW: {obj0.qc_reasons}")
                    if obj7.qc_flag == "REVIEW":
                        reasons.append(f"t7_REVIEW: {obj7.qc_reasons}")

                    if fc_v < float(self.config.fold_change_min_review):
                        reasons.append(
                            f"extreme_shrinkage (FC={fc_v:.3f} < {self.config.fold_change_min_review:.2f})"
                        )
                    elif fc_v > float(self.config.fold_change_max_review):
                        reasons.append(
                            f"extreme_expansion (FC={fc_v:.3f} > {self.config.fold_change_max_review:.2f})"
                        )

                    if not denom_pass:
                        reasons.append(
                            f"sub_denominator_baseline (d0={d0:.1f}um < {self.config.min_t0_d_um_denominator:.1f}um)"
                        )

                    if (
                        obj0.qc_flag == "REVIEW"
                        or obj7.qc_flag == "REVIEW"
                        or fc_v < float(self.config.fold_change_min_review)
                        or fc_v > float(self.config.fold_change_max_review)
                    ):
                        comb_qc = "REVIEW"
                    else:
                        comb_qc = "PASS"

                pair_id = f"{derived_pair_key}_pair{len(paired_records) + 1}"
                qc_reasons_str = "; ".join(reasons) if reasons else "None"

                rec = SpheroidPairRecord(
                    pair_id=pair_id,
                    pair_key=derived_pair_key,
                    condition=obj0.condition,
                    replicate=obj0.replicate,
                    fov=obj0.fov,
                    t0_object_id=obj0.object_id,
                    t0_centroid_x=float(obj0.centroid_x),
                    t0_centroid_y=float(obj0.centroid_y),
                    t0_eq_diameter_um=d0,
                    t0_volume_sphere_um3=v0,
                    t0_volume_ellipsoid_um3=float(obj0.volume_ellipsoid_um3),
                    t0_circularity=float(obj0.circularity),
                    t0_qc_flag=obj0.qc_flag,
                    t7_object_id=obj7.object_id,
                    t7_centroid_x=float(obj7.centroid_x),
                    t7_centroid_y=float(obj7.centroid_y),
                    t7_eq_diameter_um=d7,
                    t7_volume_sphere_um3=v7,
                    t7_volume_ellipsoid_um3=float(obj7.volume_ellipsoid_um3),
                    t7_circularity=float(obj7.circularity),
                    t7_qc_flag=obj7.qc_flag,
                    centroid_distance_px=d_px,
                    delta_volume_um3=delta_v,
                    fold_change_volume=fc_v,
                    log2_fold_change=log2_fc,
                    delta_diameter_um=delta_d,
                    fold_change_diameter=fc_d,
                    denominator_gate_pass=denom_pass,
                    combined_qc_flag=comb_qc,
                    qc_reasons=qc_reasons_str,
                )
                paired_records.append(rec)

        unpaired_t0 = [t0_objects[i] for i in range(n0) if i not in matched_t0_indices]
        unpaired_t7 = [t7_objects[j] for j in range(n7) if j not in matched_t7_indices]

        return paired_records, unpaired_t0, unpaired_t7


def pair_timepoints(
    t0_objects: List[SpheroidObjectRecord],
    t7_objects: List[SpheroidObjectRecord],
    config: Optional[Union[PairingConfig, PipelineConfig]] = None,
    pair_key: Optional[str] = None,
) -> Tuple[List[SpheroidPairRecord], List[SpheroidObjectRecord], List[SpheroidObjectRecord]]:
    """
    High-level functional interface to pair t0 and t7 objects for a given FOV.
    """
    pairer = SpheroidPairer(config=config)
    return pairer.pair_fov_objects(t0_objects, t7_objects, pair_key=pair_key)


# Alias for backward and cross-module compatibility
pair_fov_objects = pair_timepoints


def update_objects_with_pair_ids(
    objects: List[SpheroidObjectRecord],
    pairs: List[SpheroidPairRecord],
) -> List[SpheroidObjectRecord]:
    """
    Updates object records with their assigned pair_id (or 'UNPAIRED').
    Preserves all object attributes via dataclasses.replace.
    Returns a new list of updated SpheroidObjectRecord objects.
    """
    t0_to_pair = {p.t0_object_id: p.pair_id for p in pairs}
    t7_to_pair = {p.t7_object_id: p.pair_id for p in pairs}

    updated_records: List[SpheroidObjectRecord] = []
    for obj in objects:
        assigned_id = t0_to_pair.get(obj.object_id, t7_to_pair.get(obj.object_id, "UNPAIRED"))
        updated_records.append(replace(obj, pair_id=assigned_id))
    return updated_records


def pair_dataset(
    all_objects: List[SpheroidObjectRecord],
    config: Optional[Union[PairingConfig, PipelineConfig]] = None,
) -> Tuple[List[SpheroidPairRecord], List[SpheroidObjectRecord]]:
    """
    Pairs all objects across an entire multi-well dataset by grouping by pair_key.
    Returns (all_pairs, updated_objects_with_pair_ids).
    """
    pairer = SpheroidPairer(config=config)
    
    # Group objects by pair_key
    fov_groups: Dict[str, Dict[str, List[SpheroidObjectRecord]]] = {}
    for obj in all_objects:
        pk = obj.pair_key
        if pk not in fov_groups:
            fov_groups[pk] = {"t0": [], "t7": []}
        if parse_timepoint(obj.timepoint) == 0:
            fov_groups[pk]["t0"].append(obj)
        else:
            fov_groups[pk]["t7"].append(obj)

    all_pairs: List[SpheroidPairRecord] = []
    for pk, grp in sorted(fov_groups.items()):
        pairs, _, _ = pairer.pair_fov_objects(grp["t0"], grp["t7"], pair_key=pk)
        all_pairs.extend(pairs)

    updated_objects = update_objects_with_pair_ids(all_objects, all_pairs)
    return all_pairs, updated_objects


def save_pairs_csv(
    pairs: List[SpheroidPairRecord],
    output_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Saves list of SpheroidPairRecord instances to CSV adhering to the schema.
    Returns the created pandas DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [p.to_dict() for p in pairs]
    df = pd.DataFrame(rows, columns=PAIRS_CSV_COLUMNS)
    df.to_csv(output_path, index=False)
    logger.debug(f"Saved {len(pairs)} pairs to {output_path}")
    return df
