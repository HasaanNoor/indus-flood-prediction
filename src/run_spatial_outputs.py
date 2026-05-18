from clip_era5_to_sindh import clip_era5_to_sindh
from clip_glofas_to_sindh import clip_glofas_to_sindh
from clip_srtm_to_sindh import clip_srtm_to_sindh
from plot_quick_visualizations import main as plot_quick_visualizations
from plot_temporal_plots import main as plot_temporal_plots


def main() -> None:
    print("Creating clipped datasets and figures.")
    clip_era5_to_sindh()
    clip_glofas_to_sindh()
    clip_srtm_to_sindh()
    plot_quick_visualizations()
    plot_temporal_plots()
    print("\nSpatial output pipeline finished.")


if __name__ == "__main__":
    main()
