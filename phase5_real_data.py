import numpy as np
import lightkurve as lk

# Import your master function from the Phase 4 script
# (Ensure your Phase 4 file is named phase4_pipeline.py)
from phase4_pipeline import fit

def run_phase5():
    print("-> Step A: Downloading real TESS data for WASP-126...")
    # Search NASA's MAST archive for WASP-126 data (Short Cadence = 120 seconds)
    search_result = lk.search_lightcurve("WASP-126", author="SPOC", exptime=120)
    
    # Download the very first sector of data available
    lc = search_result[0].download()
    
    print("-> Step B: Preprocessing the messy space data...")
    # 1. remove_nans(): Drops missing data points
    # 2. flatten(): Removes long-term stellar variations using a rolling window
    # 3. remove_outliers(): Strips out cosmic ray hits
    clean_lc = lc.remove_nans().flatten(window_length=101).remove_outliers()
    
    # Extract the raw numbers for our pipeline
    time = clean_lc.time.value
    flux = clean_lc.flux.value
    
    # Ensure the baseline flux is exactly 1.0
    flux = flux / np.median(flux)
    
    print("-> Step C: Feeding real data into your Phase 4 pipeline...\n")
    # This calls the exact same black-box function you built!
    payload = fit(time, flux)
    
    print("\n" + "="*50)
    print("✅ FINAL DASHBOARD PAYLOAD (WASP-126 b)")
    print("="*50)
    for key, value in payload['parameters'].items():
        print(f"{key}: {value}")
        
    print("\n" + "="*50)
    print("LITERATURE COMPARISON (Did we get it right?)")
    print("="*50)
    print("Expected Period: ~3.28 days")
    print("Expected Radius: ~13.3 Earth Radii (or ~1.19 Jupiter Radii)")

if __name__ == "__main__":
    run_phase5()