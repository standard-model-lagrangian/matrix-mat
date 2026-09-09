"""Synthetic Brightfield Spheroid Image Generator for Ground-Truth Testing.

Implements a physical generative model simulating brightfield microscopy:
- Radial and planar background illumination gradients (vignetting, uneven lighting)
- Additive Gaussian sensor noise
- Dark light-absorbing cellular spheroids (circular and prolate/oblate ellipsoids)
- Touching / overlapping spheroid pairs for watershed separation validation
- Bright refractive gel condensation blobs / bubbles (contrast gate rejection)
- Sub-resolution debris specks and border-clipped spheroids
- Temporal pairs (t0 -> t7) with controlled growth rates and centroid shifts
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticObject:
  """Metadata and geometric definition for a synthetic object."""

  label_id: int
  center_x: float
  center_y: float
  radius_x: float
  radius_y: float
  angle_deg: float = 0.0
  is_dark: bool = True  # True: dark spheroid, False: bright gel blob
  contrast: float = 0.35  # Relative contrast fraction against background
  blur_sigma: float = 1.0  # Gaussian edge blur to simulate optical PSF
  name: str = "spheroid"

  @property
  def equivalent_diameter_px(self) -> float:
    """Exact ground-truth equivalent diameter in pixels: d = 2 * sqrt(rx * ry)."""
    return 2.0 * math.sqrt(self.radius_x * self.radius_y)

  @property
  def area_px(self) -> float:
    """Exact ground-truth area in pixels: A = pi * rx * ry."""
    return math.pi * self.radius_x * self.radius_y

  @property
  def volume_sphere_px3(self) -> float:
    """Exact ground-truth spherical volume: V = (pi/6) * d^3."""
    d = self.equivalent_diameter_px
    return (math.pi / 6.0) * (d**3)

  @property
  def volume_ellipsoid_px3(self) -> float:
    """Exact ground-truth ellipsoidal volume: V = (pi/6) * (2*rx) * (2*ry)^2."""
    # Assuming rx is major axis (a) and ry is minor axis (b)
    major = max(self.radius_x, self.radius_y) * 2.0
    minor = min(self.radius_x, self.radius_y) * 2.0
    return (math.pi / 6.0) * major * (minor**2)


class SyntheticImageGenerator:
  """Generates synthetic brightfield microscopy images with rigorous ground-truth."""

  def __init__(
      self,
      width: int = 800,
      height: int = 600,
      pixel_size_um: float = 1.518817,
      bg_center: float = 0.70,
      bg_edge: float = 0.55,
      noise_sigma: float = 0.02,
      seed: Optional[int] = 42,
  ):
    self.width = width
    self.height = height
    self.pixel_size_um = pixel_size_um
    self.bg_center = bg_center
    self.bg_edge = bg_edge
    self.noise_sigma = noise_sigma
    self.objects: List[SyntheticObject] = []
    self.rng = np.random.RandomState(seed)

  def add_spheroid(
      self,
      center_x: float,
      center_y: float,
      radius: float,
      contrast: float = 0.35,
      blur_sigma: float = 1.0,
      name: str = "spheroid",
  ) -> SyntheticObject:
    """Add a circular dark spheroid to the canvas."""
    label_id = len(self.objects) + 1
    obj = SyntheticObject(
        label_id=label_id,
        center_x=center_x,
        center_y=center_y,
        radius_x=radius,
        radius_y=radius,
        angle_deg=0.0,
        is_dark=True,
        contrast=contrast,
        blur_sigma=blur_sigma,
        name=name,
    )
    self.objects.append(obj)
    return obj

  def add_ellipsoid(
      self,
      center_x: float,
      center_y: float,
      radius_x: float,
      radius_y: float,
      angle_deg: float = 0.0,
      contrast: float = 0.35,
      blur_sigma: float = 1.0,
      name: str = "ellipsoid_spheroid",
  ) -> SyntheticObject:
    """Add an elliptical dark spheroid to the canvas."""
    label_id = len(self.objects) + 1
    obj = SyntheticObject(
        label_id=label_id,
        center_x=center_x,
        center_y=center_y,
        radius_x=radius_x,
        radius_y=radius_y,
        angle_deg=angle_deg,
        is_dark=True,
        contrast=contrast,
        blur_sigma=blur_sigma,
        name=name,
    )
    self.objects.append(obj)
    return obj

  def add_bright_blob(
      self,
      center_x: float,
      center_y: float,
      radius: float,
      contrast: float = 0.35,
      blur_sigma: float = 1.0,
      name: str = "bright_gel_blob",
  ) -> SyntheticObject:
    """Add a bright refractive gel blob or condensation artifact (should be rejected)."""
    label_id = len(self.objects) + 1
    obj = SyntheticObject(
        label_id=label_id,
        center_x=center_x,
        center_y=center_y,
        radius_x=radius,
        radius_y=radius,
        angle_deg=0.0,
        is_dark=False,  # Bright
        contrast=contrast,
        blur_sigma=blur_sigma,
        name=name,
    )
    self.objects.append(obj)
    return obj

  def generate_background(self) -> np.ndarray:
    """Create radial vignetting background with smooth gradient."""
    y, x = np.mgrid[: self.height, : self.width]
    cx = self.width / 2.0
    cy = self.height / 2.0
    max_r = math.sqrt(cx**2 + cy**2)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
    bg = self.bg_center - (self.bg_center - self.bg_edge) * r
    return bg.astype(np.float32)

  def render(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Render the synthetic scene into a grayscale image and ground-truth mask.

    Returns:
        image_u8: 2D uint8 numpy array [0, 255]
        ground_truth_labels: 2D int32 numpy array with instance labels (0 for bg)
        ground_truth_records: List of dictionaries detailing exact ground truth metrics
    """
    bg = self.generate_background()
    transmission = np.ones((self.height, self.width), dtype=np.float32)
    labels = np.zeros((self.height, self.width), dtype=np.int32)
    records = []

    y, x = np.mgrid[: self.height, : self.width]

    for obj in self.objects:
      # Binary disk / ellipse mask
      cos_a = math.cos(math.radians(obj.angle_deg))
      sin_a = math.sin(math.radians(obj.angle_deg))
      dx = x - obj.center_x
      dy = y - obj.center_y
      x_rot = dx * cos_a + dy * sin_a
      y_rot = -dx * sin_a + dy * cos_a

      dist_sq = (x_rot / max(obj.radius_x, 1e-4)) ** 2 + (
          y_rot / max(obj.radius_y, 1e-4)
      ) ** 2
      mask = (dist_sq <= 1.0).astype(np.float32)

      # Smooth edges to simulate optical PSF
      if obj.blur_sigma > 0:
        mask_smoothed = gaussian_filter(mask, sigma=obj.blur_sigma)
      else:
        mask_smoothed = mask

      if obj.is_dark:
        # Dark absorber: transmission decreases
        transmission *= 1.0 - obj.contrast * mask_smoothed
        # Only assign dark spheroids to ground truth positive labels
        labels[mask > 0.5] = obj.label_id
      else:
        # Bright blob: transmission increases
        transmission *= 1.0 + obj.contrast * mask_smoothed

      eq_d_um = obj.equivalent_diameter_px * self.pixel_size_um
      area_um2 = obj.area_px * (self.pixel_size_um**2)
      v_sph_um3 = (math.pi / 6.0) * (eq_d_um**3)

      records.append({
          "label_id": obj.label_id,
          "name": obj.name,
          "is_dark": obj.is_dark,
          "center_x": obj.center_x,
          "center_y": obj.center_y,
          "radius_x_px": obj.radius_x,
          "radius_y_px": obj.radius_y,
          "equivalent_diameter_px": obj.equivalent_diameter_px,
          "equivalent_diameter_um": eq_d_um,
          "area_px": obj.area_px,
          "area_um2": area_um2,
          "volume_sphere_px3": obj.volume_sphere_px3,
          "volume_sphere_um3": v_sph_um3,
          "volume_ellipsoid_px3": obj.volume_ellipsoid_px3,
      })

    # Combine background and transmission with additive sensor noise
    raw_float = bg * transmission
    if self.noise_sigma > 0:
      noise = self.rng.normal(0, self.noise_sigma, (self.height, self.width))
      raw_float += noise

    raw_clipped = np.clip(raw_float, 0.0, 1.0)
    image_u8 = (raw_clipped * 255.0).astype(np.uint8)

    return image_u8, labels, records

  # =========================================================================
  # Standardized Pre-packaged Benchmark Scenarios
  # =========================================================================

  @classmethod
  def create_isolated_spheroid(
      cls,
      diameter_px: float = 100.0,
      width: int = 800,
      height: int = 600,
      pixel_size_um: float = 1.518817,
      contrast: float = 0.35,
      noise_sigma: float = 0.02,
  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Scenario 1: Isolated dark circular spheroid."""
    gen = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=noise_sigma,
    )
    radius = diameter_px / 2.0
    obj = gen.add_spheroid(
        center_x=width / 2.0,
        center_y=height / 2.0,
        radius=radius,
        contrast=contrast,
        blur_sigma=1.0,
        name="isolated_spheroid",
    )
    img, labels, recs = gen.render()
    return img, labels, recs[0]

  @classmethod
  def create_touching_pair(
      cls,
      d1_px: float = 80.0,
      d2_px: float = 80.0,
      overlap_px: float = 10.0,
      width: int = 800,
      height: int = 600,
      pixel_size_um: float = 1.518817,
      contrast: float = 0.35,
      noise_sigma: float = 0.02,
  ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Scenario 2: Touching pair of spheroids forming a dumbbell."""
    gen = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=noise_sigma,
    )
    r1 = d1_px / 2.0
    r2 = d2_px / 2.0
    # Separation distance between centers
    distance = r1 + r2 - overlap_px
    cy = height / 2.0
    cx1 = (width / 2.0) - (distance / 2.0)
    cx2 = (width / 2.0) + (distance / 2.0)

    gen.add_spheroid(
        center_x=cx1,
        center_y=cy,
        radius=r1,
        contrast=contrast,
        blur_sigma=1.0,
        name="touching_spheroid_1",
    )
    gen.add_spheroid(
        center_x=cx2,
        center_y=cy,
        radius=r2,
        contrast=contrast,
        blur_sigma=1.0,
        name="touching_spheroid_2",
    )

    img, labels, recs = gen.render()
    return img, labels, recs

  @classmethod
  def create_bright_blob(
      cls,
      diameter_px: float = 120.0,
      width: int = 800,
      height: int = 600,
      pixel_size_um: float = 1.518817,
      contrast: float = 0.35,
      noise_sigma: float = 0.02,
  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Scenario 3: Bright refractive gel condensation blob (must yield 0 masks)."""
    gen = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=noise_sigma,
    )
    radius = diameter_px / 2.0
    gen.add_bright_blob(
        center_x=width / 2.0,
        center_y=height / 2.0,
        radius=radius,
        contrast=contrast,
        blur_sigma=1.0,
        name="bright_gel_blob",
    )
    img, labels, recs = gen.render()
    return img, labels, recs[0]

  @classmethod
  def create_multi_spheroid_field(
      cls,
      n_spheroids: int = 8,
      width: int = 1024,
      height: int = 768,
      pixel_size_um: float = 1.518817,
      include_debris: bool = True,
      include_border: bool = True,
      include_bright_blob: bool = True,
      seed: int = 101,
  ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Complex multi-spheroid field with mix of valid spheroids, debris, border objects, and blobs."""
    gen = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=0.02,
        seed=seed,
    )

    # Valid internal spheroids (d: 100 - 300 um => radius: 33 - 100 px)
    for i in range(n_spheroids):
      cx = gen.rng.uniform(100, width - 100)
      cy = gen.rng.uniform(100, height - 100)
      r = gen.rng.uniform(35, 75)
      # Randomly elliptical
      is_ellip = gen.rng.rand() > 0.5
      if is_ellip:
        rx = r * gen.rng.uniform(0.8, 1.2)
        ry = r * gen.rng.uniform(0.8, 1.2)
        gen.add_ellipsoid(
            cx, cy, rx, ry, angle_deg=gen.rng.uniform(0, 180), contrast=0.35
        )
      else:
        gen.add_spheroid(cx, cy, r, contrast=0.35)

    if include_debris:
      # Debris (<40 um => radius < 13 px)
      for _ in range(3):
        cx = gen.rng.uniform(50, width - 50)
        cy = gen.rng.uniform(50, height - 50)
        gen.add_spheroid(cx, cy, radius=8.0, contrast=0.25, name="debris")

    if include_border:
      # Border-touching object (within 15px of border)
      gen.add_spheroid(
          center_x=8.0,
          center_y=height / 2.0,
          radius=40.0,
          name="border_touching_left",
      )

    if include_bright_blob:
      # Bright gel blob
      gen.add_bright_blob(
          center_x=width / 2.0,
          center_y=height / 2.0,
          radius=50.0,
          name="refractive_blob",
      )

    img, labels, recs = gen.render()
    return img, labels, recs

  @classmethod
  def create_temporal_pair(
      cls,
      n_spheroids: int = 5,
      growth_factors: Optional[List[float]] = None,
      max_drift_px: float = 15.0,
      width: int = 800,
      height: int = 600,
      pixel_size_um: float = 1.518817,
      seed: int = 202,
  ) -> Tuple[
      Tuple[np.ndarray, np.ndarray],
      Tuple[np.ndarray, np.ndarray],
      List[Dict[str, Any]],
  ]:
    """Generate aligned Day 0 and Day 7 image pair with known ground truth fold changes."""
    if growth_factors is None:
      # Fold changes in volume: 1.5x, 2.0x, 0.8x, 3.0x, 1.2x
      growth_factors = [1.5, 2.0, 0.8, 3.0, 1.2]

    gen0 = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=0.02,
        seed=seed,
    )
    gen7 = cls(
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        noise_sigma=0.02,
        seed=seed + 1,
    )

    pair_truth = []
    for i, fc_vol in enumerate(growth_factors[:n_spheroids]):
      cx0 = gen0.rng.uniform(120, width - 120)
      cy0 = gen0.rng.uniform(120, height - 120)
      # Radius for d0 ~ 120 um => r0 ~ 40 px
      r0 = gen0.rng.uniform(35, 50)

      # Volume scaling: V7 = V0 * fc => d7 = d0 * (fc)^(1/3) => r7 = r0 * (fc)^(1/3)
      radius_factor = fc_vol ** (1.0 / 3.0)
      r7 = r0 * radius_factor

      # Slight translation drift between timepoints
      drift_x = gen0.rng.uniform(-max_drift_px, max_drift_px)
      drift_y = gen0.rng.uniform(-max_drift_px, max_drift_px)
      cx7 = cx0 + drift_x
      cy7 = cy0 + drift_y

      gen0.add_spheroid(cx0, cy0, r0, name=f"pair_{i+1}_t0")
      gen7.add_spheroid(cx7, cy7, r7, name=f"pair_{i+1}_t7")

      d0_um = 2.0 * r0 * pixel_size_um
      d7_um = 2.0 * r7 * pixel_size_um
      v0_um3 = (math.pi / 6.0) * (d0_um**3)
      v7_um3 = (math.pi / 6.0) * (d7_um**3)

      pair_truth.append({
          "pair_idx": i + 1,
          "t0_center": (cx0, cy0),
          "t7_center": (cx7, cy7),
          "drift_distance_px": math.hypot(drift_x, drift_y),
          "t0_diameter_um": d0_um,
          "t7_diameter_um": d7_um,
          "t0_volume_um3": v0_um3,
          "t7_volume_um3": v7_um3,
          "true_volume_fold_change": fc_vol,
          "denominator_pass": d0_um >= 60.0,
      })

    img0, lbl0, _ = gen0.render()
    img7, lbl7, _ = gen7.render()

    return (img0, lbl0), (img7, lbl7), pair_truth
