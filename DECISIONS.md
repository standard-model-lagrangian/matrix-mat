# Pipeline Architectural & Fallback Decisions Log

This document records all runtime segmentation backend attempts, plausibility rejections, fallbacks, and parameter tuning events.

---

### [2026-08-31 21:40:06Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:06Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:07Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:07Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:07Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:08Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:08Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:08Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:08Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:15Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0002_TRANS`
- **Image ID**: `Day0 Mat gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0002_TRANS`
- **Image ID**: `Day0 Mat gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:17Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0002_TRANS`
- **Image ID**: `Day0 Mat gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 7 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 7

---

### [2026-08-31 21:40:17Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0003_TRANS`
- **Image ID**: `Day0 Mat gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:17Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0003_TRANS`
- **Image ID**: `Day0 Mat gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:21Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0003_TRANS`
- **Image ID**: `Day0 Mat gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 10 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 10

---

### [2026-08-31 21:40:21Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0001_TRANS`
- **Image ID**: `Day7 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:21Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0001_TRANS`
- **Image ID**: `Day7 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:23Z] Segmentation Hierarchy Result — `Day7 Mat gel1_0001_TRANS`
- **Image ID**: `Day7 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 41 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 41

---

### [2026-08-31 21:40:23Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0002_TRANS`
- **Image ID**: `Day7 Mat gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:23Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0002_TRANS`
- **Image ID**: `Day7 Mat gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:26Z] Segmentation Hierarchy Result — `Day7 Mat gel1_0002_TRANS`
- **Image ID**: `Day7 Mat gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 50 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 50

---

### [2026-08-31 21:40:26Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0003_TRANS`
- **Image ID**: `Day7 Mat gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:26Z] Segmentation Hierarchy Fallback — `Day7 Mat gel1_0003_TRANS`
- **Image ID**: `Day7 Mat gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:28Z] Segmentation Hierarchy Result — `Day7 Mat gel1_0003_TRANS`
- **Image ID**: `Day7 Mat gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 28 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 28

---

### [2026-08-31 21:40:28Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:28Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:31Z] Segmentation Hierarchy Result — `Day0 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 76 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 76

---

### [2026-08-31 21:40:31Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:31Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:34Z] Segmentation Hierarchy Result — `Day0 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 44 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 44

---

### [2026-08-31 21:40:34Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:34Z] Segmentation Hierarchy Fallback — `Day0 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:37Z] Segmentation Hierarchy Result — `Day0 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 78 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 78

---

### [2026-08-31 21:40:37Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:37Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:40Z] Segmentation Hierarchy Result — `Day7 S34D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 25 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 25

---

### [2026-08-31 21:40:40Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:40Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:43Z] Segmentation Hierarchy Result — `Day7 S34D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 14 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 14

---

### [2026-08-31 21:40:43Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:43Z] Segmentation Hierarchy Fallback — `Day7 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Result — `Day7 S34D30 gel1_0003_TRANS`
- **Image ID**: `Day7 S34D30 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 18 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 18

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:45Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:46Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:46Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:46Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:46Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:40:47Z] Segmentation Hierarchy Result — `Day0 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 11 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 11

---

### [2026-08-31 21:40:47Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:47Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:48Z] Segmentation Hierarchy Result — `Day0 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 20 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 20

---

### [2026-08-31 21:40:48Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:48Z] Segmentation Hierarchy Fallback — `Day0 S40D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:52Z] Segmentation Hierarchy Result — `Day0 S40D30 gel1_0003_TRANS`
- **Image ID**: `Day0 S40D30 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 19 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 19

---

### [2026-08-31 21:40:52Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:52Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:55Z] Segmentation Hierarchy Result — `Day7 S40D30 gel1_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 22 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 22

---

### [2026-08-31 21:40:55Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:55Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:56Z] Segmentation Hierarchy Result — `Day7 S40D30 gel1_0002_TRANS`
- **Image ID**: `Day7 S40D30 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 56 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 56

---

### [2026-08-31 21:40:56Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel2_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel2_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:56Z] Segmentation Hierarchy Fallback — `Day7 S40D30 gel2_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel2_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:58Z] Segmentation Hierarchy Result — `Day7 S40D30 gel2_0001_TRANS`
- **Image ID**: `Day7 S40D30 gel2_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 19 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 19

---

### [2026-08-31 21:40:58Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:58Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:59Z] Segmentation Hierarchy Result — `Day0 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 26 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 26

---

### [2026-08-31 21:40:59Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:40:59Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:00Z] Segmentation Hierarchy Result — `Day0 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 23 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 23

---

### [2026-08-31 21:41:00Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:00Z] Segmentation Hierarchy Fallback — `Day0 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:01Z] Segmentation Hierarchy Result — `Day0 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day0 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 8 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 8

---

### [2026-08-31 21:41:01Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:01Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:03Z] Segmentation Hierarchy Result — `Day7 S43D20 gel1_0001_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 30 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 30

---

### [2026-08-31 21:41:03Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:03Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:05Z] Segmentation Hierarchy Result — `Day7 S43D20 gel1_0002_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 22 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 22

---

### [2026-08-31 21:41:06Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:06Z] Segmentation Hierarchy Fallback — `Day7 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:08Z] Segmentation Hierarchy Result — `Day7 S43D20 gel1_0003_TRANS`
- **Image ID**: `Day7 S43D20 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 23 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 23

---

### [2026-08-31 21:41:08Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:08Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:10Z] Segmentation Hierarchy Result — `Day0 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 78 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 78

---

### [2026-08-31 21:41:10Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:10Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:13Z] Segmentation Hierarchy Result — `Day0 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 18 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 18

---

### [2026-08-31 21:41:13Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:13Z] Segmentation Hierarchy Fallback — `Day0 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:14Z] Segmentation Hierarchy Result — `Day0 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day0 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 31 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 31

---

### [2026-08-31 21:41:14Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:14Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:14Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:14Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Result — `Day7 S46D10 gel1_0001_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 31 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 31

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:15Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:16Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:41:17Z] Segmentation Hierarchy Result — `Day7 S46D10 gel1_0002_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 38 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 38

---

### [2026-08-31 21:41:17Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:17Z] Segmentation Hierarchy Fallback — `Day7 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:19Z] Segmentation Hierarchy Result — `Day7 S46D10 gel1_0003_TRANS`
- **Image ID**: `Day7 S46D10 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 29 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 29

---

### [2026-08-31 21:41:19Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0001_TRANS`
- **Image ID**: `Day0 S50 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:19Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0001_TRANS`
- **Image ID**: `Day0 S50 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:20Z] Segmentation Hierarchy Result — `Day0 S50 gel1_0001_TRANS`
- **Image ID**: `Day0 S50 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 7 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 7

---

### [2026-08-31 21:41:21Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0002_TRANS`
- **Image ID**: `Day0 S50 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:21Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0002_TRANS`
- **Image ID**: `Day0 S50 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:23Z] Segmentation Hierarchy Result — `Day0 S50 gel1_0002_TRANS`
- **Image ID**: `Day0 S50 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 64 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 64

---

### [2026-08-31 21:41:23Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0003_TRANS`
- **Image ID**: `Day0 S50 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:23Z] Segmentation Hierarchy Fallback — `Day0 S50 gel1_0003_TRANS`
- **Image ID**: `Day0 S50 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:24Z] Segmentation Hierarchy Result — `Day0 S50 gel1_0003_TRANS`
- **Image ID**: `Day0 S50 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 23 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 23

---

### [2026-08-31 21:41:24Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0001_TRANS`
- **Image ID**: `Day7 S50 gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:24Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0001_TRANS`
- **Image ID**: `Day7 S50 gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:26Z] Segmentation Hierarchy Result — `Day7 S50 gel1_0001_TRANS`
- **Image ID**: `Day7 S50 gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 33 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 33

---

### [2026-08-31 21:41:26Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0002_TRANS`
- **Image ID**: `Day7 S50 gel1_0002_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:26Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0002_TRANS`
- **Image ID**: `Day7 S50 gel1_0002_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:28Z] Segmentation Hierarchy Result — `Day7 S50 gel1_0002_TRANS`
- **Image ID**: `Day7 S50 gel1_0002_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 36 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 36

---

### [2026-08-31 21:41:28Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0003_TRANS`
- **Image ID**: `Day7 S50 gel1_0003_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:28Z] Segmentation Hierarchy Fallback — `Day7 S50 gel1_0003_TRANS`
- **Image ID**: `Day7 S50 gel1_0003_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:41:32Z] Segmentation Hierarchy Result — `Day7 S50 gel1_0003_TRANS`
- **Image ID**: `Day7 S50 gel1_0003_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 88 objects identified
- **Details**:
  - `pixel_size_um`: 1.5188172690164
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 88

---

### [2026-08-31 21:42:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:42:42Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:48:38Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:38Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:39Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:48:40Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:53:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:40Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:53:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:40Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:41Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:53:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:41Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:53:41Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_isolated`
- **Image ID**: `synthetic_isolated`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_isolated`
- **Image ID**: `synthetic_isolated`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Result — `synthetic_isolated`
- **Image ID**: `synthetic_isolated`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 1

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_touching`
- **Image ID**: `synthetic_touching`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_touching`
- **Image ID**: `synthetic_touching`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Result — `synthetic_touching`
- **Image ID**: `synthetic_touching`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 2 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 2

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_bright_blob`
- **Image ID**: `synthetic_bright_blob`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Fallback — `synthetic_bright_blob`
- **Image ID**: `synthetic_bright_blob`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:58:09Z] Segmentation Hierarchy Result — `synthetic_bright_blob`
- **Image ID**: `synthetic_bright_blob`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 0 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 0

---

### [2026-08-31 21:59:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:45Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:59:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:46Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 21:59:46Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:46Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 21:59:46Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:16:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:48Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:16:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:49Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:16:49Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:49Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:16:50Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:19:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:45Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:47Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:19:47Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:47Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:48Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-08-31 22:19:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:48Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-08-31 22:19:49Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:27:00Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:00Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:01Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:27:02Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:28:02Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:02Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:02Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:28:02Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:02Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:03Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:28:03Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:03Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:28:03Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:32:11Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:11Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Cellpose-SAM plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cpsam` to `cyto3`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:12Z] Segmentation Hierarchy Fallback — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `cyto3`
- **Trigger / Event**: Cellpose cyto3 plausibility check failed: 0 objects found
- **Action Taken**: Switched from `cyto3` to `classical`
- **Outcome**: Fallback backend engaged

---

### [2026-09-01 01:32:13Z] Segmentation Hierarchy Result — `Day0 Mat gel1_0001_TRANS`
- **Image ID**: `Day0 Mat gel1_0001_TRANS`
- **Backend Attempted**: `classical`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `classical` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 2048x1536
  - `objects_found`: 1

---

### [2026-09-03 06:08:23Z] Segmentation Hierarchy Result — `synthetic_isolated`
- **Image ID**: `synthetic_isolated`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 1

---

### [2026-09-03 06:08:24Z] Segmentation Hierarchy Result — `synthetic_touching`
- **Image ID**: `synthetic_touching`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 2 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 2

---

### [2026-09-03 06:08:24Z] Segmentation Hierarchy Result — `synthetic_bright_blob`
- **Image ID**: `synthetic_bright_blob`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 1

---

### [2026-09-03 07:26:32Z] Segmentation Hierarchy Result — `synthetic_isolated`
- **Image ID**: `synthetic_isolated`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 1

---

### [2026-09-03 07:26:32Z] Segmentation Hierarchy Result — `synthetic_touching`
- **Image ID**: `synthetic_touching`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 2 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 2

---

### [2026-09-03 07:26:33Z] Segmentation Hierarchy Result — `synthetic_bright_blob`
- **Image ID**: `synthetic_bright_blob`
- **Backend Attempted**: `cpsam`
- **Trigger / Event**: Segmentation completed
- **Action Taken**: Segmented with `cpsam` backend
- **Outcome**: SUCCESS: 1 objects identified
- **Details**:
  - `pixel_size_um`: 1.518817
  - `scale_factor`: 1.0
  - `raw_dimensions`: 800x600
  - `objects_found`: 1

---

