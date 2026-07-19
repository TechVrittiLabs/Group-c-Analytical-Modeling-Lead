import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

# ==========================================
# STEP 5: Create a Mock Light Curve Generator
# (টেস্ট করার জন্য কৃত্রিম বা ফেক ডেটা তৈরি করা)
# ==========================================
def generate_mock_lightcurve(period, depth, duration, noise_level, days=20, points_per_day=24):
    """
    নয়েজসহ একটি ফেক লাইট কার্ভ তৈরি করে।
    """
    # 20 দিনের ডেটা তৈরি করছি
    time = np.linspace(0, days, days * points_per_day)
    flux = np.ones_like(time) # প্রাথমিক আলো 1.0 ধরে নিচ্ছি
    
    # গ্রহের ট্রানজিট বা আলো কমে যাওয়ার সময়গুলো হিসাব করছি
    transit_times = time % period
    in_transit = (transit_times < (duration / 2)) | (transit_times > (period - duration / 2))
    
    # যে সময়গুলোতে গ্রহ সামনে থাকবে, সেগুলোর আলো (depth) কমিয়ে দিচ্ছি
    flux[in_transit] -= depth
    
    # ডেটায় র্যান্ডম নয়েজ (Gaussian noise) যোগ করছি
    noise = np.random.normal(0, noise_level, size=len(time))
    noisy_flux = flux + noise
    
    return time, noisy_flux

# ==========================================
# STEP 4: Build the BLS Module
# (BLS অ্যালগরিদম ব্যবহার করে সিগন্যাল খোঁজা)
# ==========================================
def find_transit_bls(time, flux):
    """
    BLS ব্যবহার করে ডেটা থেকে গ্রহের পিরিয়ড এবং ডেপথ বের করে।
    """
    model = BoxLeastSquares(time, flux)
    
    # 0.5 থেকে 20 দিনের মধ্যে পিরিয়ড খুঁজবো
    period_grid = np.linspace(0.5, 20.0, 1000) 
    duration_grid = np.linspace(0.01, 0.2, 50)
    
    results = model.power(period_grid, duration_grid)
    
    # সবচেয়ে শক্তিশালী সিগন্যাল (Maximum Power) বের করা
    best_index = np.argmax(results.power)
    best_period = results.period[best_index]
    best_depth = results.depth[best_index]
    
    return best_period, best_depth

# ==========================================
# Phase 2 Validation Test
# ==========================================
def test_phase2():
    print("১. ফেক ডেটা (Mock Light curve) তৈরি করা হচ্ছে...")
    # আমরা এই মানগুলো আগে থেকেই জানি
    true_period = 4.05
    true_depth = 0.009
    true_duration = 0.1
    
    time, flux = generate_mock_lightcurve(
        period=true_period, 
        depth=true_depth, 
        duration=true_duration, 
        noise_level=0.002 # নয়েজ যোগ করা হলো
    )
    
    print("২. BLS ব্যবহার করে সিগন্যাল খোঁজা হচ্ছে...")
    calc_period, calc_depth = find_transit_bls(time, flux)
    
    print("-" * 40)
    print(f"আসল পিরিয়ড (True Period): {true_period} দিন | হিসাব করা পিরিয়ড: {calc_period:.2f} দিন")
    print(f"আসল ডেপথ (True Depth): {true_depth} | হিসাব করা ডেপথ: {calc_depth:.4f}")
    print("-" * 40)
    
    # চেক করা হচ্ছে অ্যালগরিদম ঠিকমতো পিরিয়ড ধরতে পারল কি না
    assert np.isclose(calc_period, true_period, atol=0.1), "BLS অ্যালগরিদম সিগন্যাল ধরতে ব্যর্থ হয়েছে!"
    print("✅ Phase 2 Test Passed: BLS মডিউল নয়েজ থেকে সফলভাবে সিগন্যাল উদ্ধার করেছে।")

if __name__ == "__main__":
    test_phase2()