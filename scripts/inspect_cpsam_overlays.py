import os, cv2, tifffile, torch
import numpy as np
from pathlib import Path
from cellpose import models

out_dir = Path("output/cpsam_tuning_inspection")
out_dir.mkdir(parents=True, exist_ok=True)

local_model = "models/cpsam_v2"
use_gpu = torch.backends.mps.is_available()
print("Loading Cellpose-SAM (cpsam_v2) on MPS:", use_gpu)
model = models.CellposeModel(pretrained_model=local_model, gpu=use_gpu)

test_imgs = [
    "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 Mat gel1_0004_TRANS.tif",
    "Experimental data /Chuling cells/Spheroid D7 260812/Day7 Mat gel1_0004_TRANS.tif",
    "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 S34D30 gel1_0001_TRANS.tif",
    "Experimental data /Chuling cells/Spheroid D7 260812/Day7 S34D30 gel1_0001_TRANS.tif",
    "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 S40D30 gel5_0001_TRANS.tif",
    "Experimental data /Chuling cells/Spheroid D7 260812/Day7 S40D30 gel5_0001_TRANS.tif",
]

for fpath in test_imgs:
    p = Path(fpath)
    if not p.exists():
        continue
    img = tifffile.imread(fpath)
    h, w = img.shape[:2]
    
    # Run CPSAM evaluation
    # Optimal parameters: diameter=160.0, flow_threshold=0.4, cellprob_threshold=0.0
    masks, flows, styles = model.eval(
        img,
        diameter=160.0,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=40,
    )
    
    n_objs = int(np.max(masks))
    print(f"[{p.name}] Segmented {n_objs} objects")
    
    # Create RGB supervision overlay
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    pixel_size_um = 1.518817
    
    for i in range(1, n_objs + 1):
        m = (masks == i).astype(np.uint8)
        area_px = float(np.sum(m))
        d_um = 2.0 * np.sqrt(area_px / np.pi) * pixel_size_um
        
        # Draw green contour
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (46, 204, 113), 2, cv2.LINE_AA)
        
        # Centroid
        M = cv2.moments(m)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(bgr, (cx, cy), 4, (46, 204, 113), -1)
            cv2.putText(bgr, f"#{i}: {d_um:.0f}um", (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, f"#{i}: {d_um:.0f}um", (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1, cv2.LINE_AA)
            
    # Add title banner
    cv2.putText(bgr, f"Cellpose-SAM (cpsam_v2) | {p.stem} | Found: {n_objs} spheroids", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, f"Cellpose-SAM (cpsam_v2) | {p.stem} | Found: {n_objs} spheroids", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
    
    save_path = out_dir / f"{p.stem}_cpsam_overlay.png"
    cv2.imwrite(str(save_path), bgr)
    print(f"Saved overlay: {save_path}")

print("Done creating test overlays.")
