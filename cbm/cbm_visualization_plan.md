# CBM Visualization Platform: Architecture & Implementation Plan

The objective of this platform is to build an interactive, web-based specific visualizer that proves the interpretability of our Concept Bottleneck Model (CBM). Standard autonomous driving visualizers only show the *actions* of the car. Our platform will provide "X-Ray Vision" into the mind of the AI, showing the *reasoning* via a real-time "Neural Indicator" dashboard.

---

## 1. The Strategy: Honest Storytelling
For the thesis presentation, the visualizer serves two purposes:
1. **The "Genius AI" Demo:** Proving that the model structurally understands the world by showing concept neurons (like `traffic_light_red`) lighting up exactly when the car approaches an intersection.
2. **The "Why it Failed" Demo:** Showing a scenario where the car crashes, and pointing to the dashboard to prove *why* it crashed (e.g., the `dist_nearest_object` concept failed to spike). This proves the value of CBM transparency over a black-box AI.

---

## 2. Phase A: Golden Scenario Curation (Backend)
We cannot use random, noisy scenarios for a high-stakes presentation. We must mathematically filter the dataset for the clearest examples.

**The Pipeline:**
1. **Mass Evaluation:** We will write a Python script (`curate_demo_scenarios.py`) that runs the trained CBM on 1,000 Waymax scenarios.
2. **The Filtering Algorithm:** For each scenario, we calculate a "Clarity Score":
   - **No Collisions:** The agent must successfully complete the route.
   - **High Concept Accuracy:** The regression R² for that specific scenario must be > 0.90, and binary concept accuracy must be 100%.
3. **Selection:** 
   - We pick the **Top 3 "Golden Scenarios"** (perfect driving + perfect concept alignment).
   - We pick **1 "Transparent Failure"** scenario (the agent crashes because a specific concept visibly failed to activate).

---

## 3. Phase B: Data Extraction Hook
Once the Golden Scenarios are selected, we need to export them for the web platform.
We will modify the rollout loop to save a JSON file per frame with the following structure:

```json
{
  "frame_id": 42,
  "ego_position": {"x": 105.2, "y": 44.1, "yaw": 1.57},
  "route_points": [...],
  "agent_positions": [...],
  "cbm_concepts": {
    "traffic_light_red": 0.98,
    "at_intersection": 0.85,
    "dist_nearest_object": 0.22,
    "ego_speed": 0.55
  }
}
```

---

## 4. Phase C: The Web Platform (Frontend)
The web interface will be split into two synchronized halves.

### Top Half: The Simulation Canvas
- A standard 2D top-down view built with HTML5 Canvas or WebGL (e.g., Three.js/PixiJS).
- Shows the ego vehicle, the map routes, and dynamic external agents plotting along their trajectories.

### Bottom Half: The "Neural Dashboard"
This is the heart of the demo. It displays the `cbm_concepts` in real-time.
- **Binary Indicators (LED Style):** Concepts like `traffic_light_red` or `lead_vehicle_decelerating` are displayed as circular indicators that visually "snap" on (glow bright red/green) when the raw value crosses a 0.5 threshold.
- **Continuous Indicators (Progress Bars):** Concepts like `ego_speed` or `curvature` are displayed as horizontal glowing bars that fill and empty dynamically.

### Crucial UI Polish: Signal Smoothing
Raw neural network predictions vibrate heavily frame-by-frame (e.g., `0.9` → `0.3` → `0.8`). If piped directly to the UI, the bars will flicker violently and look broken. 
**Solution:** We will apply an Exponential Moving Average (EMA) or a simple low-pass filter to the `c` vector arrays in Javascript before rendering. This ensures the bars glide smoothly up and down, making the AI look "confident" and polished.

---

## 5. Next Steps for Implementation
1. Write the `curate_demo_scenarios.py` script to find the Golden Scenarios using our `eval_cbm.py` loop.
2. Build the JSON export hook to save `(Traj, Concepts)`.
3. Scaffold a basic React/Vite app to load the JSON files and build the synchronous canvas + dashboard layout.
