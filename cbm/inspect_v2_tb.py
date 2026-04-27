import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def inspect_tb(log_dir):
    print(f"Inspecting TensorBoard logs in: {log_dir}")
    
    # Load the event accumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # List all available tags
    tags = ea.Tags()['scalars']
    print(f"Found {len(tags)} scalar tags.")
    
    # Metrics to extract
    target_tags = [
        'train/concept_loss',
        'train/policy_loss',
        'train/value_loss',
        'metrics/at_fault_collision',
        'metrics/accuracy',
        'metrics/progress_ratio_nuplan'
    ]
    
    report = []
    for tag in target_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            first_val = events[0].value
            last_val = events[-1].value
            steps = events[-1].step
            
            trend = "DECREASING" if last_val < first_val else "INCREASING"
            if "accuracy" in tag or "progress" in tag:
                trend = "IMPROVING" if last_val > first_val else "DECLINING"
            elif "loss" in tag:
                trend = "IMPROVING (decreasing)" if last_val < first_val else "WORSENING (increasing)"

            report.append({
                "Metric": tag,
                "Start": round(first_val, 5),
                "End": round(last_val, 5),
                "Trend": trend,
                "Steps": steps
            })
    
    # Display table
    df = pd.DataFrame(report)
    print("\nTraining Progress Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    log_dir = "cbm_v2_frozen_womd_150gb/tb"
    inspect_tb(log_dir)
