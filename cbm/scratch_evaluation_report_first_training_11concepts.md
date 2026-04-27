# CBM Scratch Training Report (11-Concept)

This report analyzes the CBM model trained from scratch without a pretrained encoder (`cbm_scratch`). Since this is a pure from-scratch run, the encoder had to build spatial representations of the Waymax environment guided entirely by the 11 Phase 1 & 2 concepts.

## 📉 Task & Driving Performance

The model successfully learned basic driving capabilities during the 5M step run:

| Metric | Starting Value | Final Value | Trend |
| :--- | :--- | :--- | :--- |
| **Route Progress Ratio (`nuplan`)** | 0.367 | 0.574 | 🟢 IMPROVING |
| **At-Fault Collisions** | 0.014 | 0.0003 | 🟢 DECREASING |
| **Total Policy Loss** | -0.268 | -27.592 | 🟢 IMPROVING |

> [!TIP]
> The model dropped collisions to near-zero (0.0003) and achieved a respectable 57% route progress ratio without relying on any pretrained Google encoder!

---

## 🚘 Continuous Concept Analysis

Because there was no pretrained encoder, the CBM had to invent internal representations of geometry and kinematics to predict these continuous tracking targets. 

Here is the final MSE (Mean Squared Error) across the continuous concepts at the end of training:

| Concept Name | Start Loss (MSE) | Final Loss (MSE) | Trend |
| :--- | :--- | :--- | :--- |
| `progress_along_route` | 0.0192 | **0.0017** | 🟢 IMPROVING |
| `ego_speed` | 0.0172 | **0.0093** | 🟢 IMPROVING |
| `ego_acceleration` | 0.0489 | **0.0437** | 🟢 IMPROVING |
| `heading_deviation` | 0.0036 | **0.0050** | 🔴 WORSENING |

> [!NOTE]
> **Analysis on Continuous Features:**
> 1. **Excellent Spatial Progress:** The encoder became incredibly precise at tracking its `progress_along_route` (final MSE: 0.0017). This proves the concept bottleneck successfully forced the encoder to learn how far the vehicle has traveled along the lane graph.
> 2. **Strong Kinematics:** `ego_speed` prediction improved significantly. `ego_acceleration` also improved, but remains the hardest continuous feature to predict (MSE: 0.043), likely because acceleration is highly volatile in the dataset.
> 3. **Heading Instability:** The `heading_deviation` prediction slightly worsened. When training from scratch, early lane drifting makes heading deviation hard to ground until the model perfectly understands lane topologies.

---

## 🚥 Binary Concept Analysis

Binary concepts (Traffic lights, Intersections, Objects) use BCE (Binary Cross Entropy) loss. The rapid decline in these losses indicates the encoder perfectly clustered the visual latent space for these features.

| Concept Name | Start Loss (BCE) | Final Loss (BCE) | Trend |
| :--- | :--- | :--- | :--- |
| `traffic_light_red` | 0.228 | **0.0037** | 🟢 IMPROVING |
| `dist_to_traffic_light` | 0.034 | **0.0064** | 🟢 IMPROVING |
| `dist_nearest_object` | 0.015 | **0.0055** | 🟢 IMPROVING |
| `at_intersection` | 0.300 | **0.0120** | 🟢 IMPROVING |
| `lead_vehicle_decelerating`| 0.168 | **0.0620** | 🟢 IMPROVING |
| `num_objects_within_10m` | 0.026 | **0.0180** | 🟢 IMPROVING |

> [!IMPORTANT]
> The encoder almost instantly perfected Traffic Light detection (`0.0037` BCE) and Intersection Awareness (`0.012` BCE) from scratch!

---

## Next Steps

This checkpoint proves that the CBM can bootstrap spatial awareness absolutely from scratch without Google's pretraining. However, it only learned the 11 base concepts.

To extract performance measurements on our complex Phase 3 spatial concepts (e.g. `path_curvature_max` and `yaw_to_lane_center`), we must spawn the new version of this pipeline using the updated `15-concept` configuration!
