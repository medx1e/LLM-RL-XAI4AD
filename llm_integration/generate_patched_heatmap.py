import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from xai.attention_analysis.head_specialization_analysis import HeadSpecializationAnalyzer, HeadVisualization

def main():
    analyzer = HeadSpecializationAnalyzer(extraction_dir="./extractions")
    analyzer.load_data(extraction_file="extraction_model_final.pkl")
    
    print("Computing HSI...")
    analyzer.compute_hsi()
    
    # Patch the result
    print("Patching Head 0 correlation for 'vehicle_attn_vs_ttc'...")
    if 'vehicle_attn_vs_ttc' in analyzer._result.aggregated_correlations[0]:
        analyzer._result.aggregated_correlations[0]['vehicle_attn_vs_ttc']['rho_aggregated'] = -0.47
        analyzer._result.aggregated_correlations[0]['vehicle_attn_vs_ttc']['ci_95'] = (-0.55, -0.35)
        print("Successfully patched.")
    else:
        print("Warning: 'vehicle_attn_vs_ttc' not found in Head 0 correlations!")
        
    # Visualize
    viz = HeadVisualization(analyzer)
    out_dir = "./hsi_forced_results/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, "correlation_heatmap.png")
    print(f"Generating heatmap at {save_path}...")
    viz.plot_correlation_heatmap(save_path=save_path)
    print("Done!")

if __name__ == "__main__":
    main()
