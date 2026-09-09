import cv2, tifffile, torch
import numpy as np
from pathlib import Path
from cellpose import models
import glob

local_model = "models/cpsam_v2"
use_gpu = torch.backends.mps.is_available()
model = models.CellposeModel(pretrained_model=local_model, gpu=use_gpu)

t0_files = sorted(glob.glob("Experimental data /Chuling cells/Spheroid Day0 260805/*.tif"))
t7_files = sorted(glob.glob("Experimental data /Chuling cells/Spheroid D7 260812/*.tif"))

# Sample 15 images from Day 0 and 15 from Day 7
sample_files = t0_files[::10][:15] + t7_files[::10][:15]

results = []
for f in sample_files:
    img = tifffile.imread(f)
    masks, flows, styles = model.eval(
        img,
        diameter=160.0,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=30,
    )
    n = int(np.max(masks))
    results.append((Path(f).name, n))
    print(f"{Path(f).name:<40} -> {n} spheroids")

zero_counts = sum(1 for _, n in results if n == 0)
print(f"\nTotal tested: {len(results)}, Zero count FOVs: {zero_counts}")
