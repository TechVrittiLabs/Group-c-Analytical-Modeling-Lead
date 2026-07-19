import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import astropy.constants as const

# JAX-এর 64-bit precision (বেশি নির্ভুলতার জন্য)
jax.config.update("jax_enable_x64", True)

# ==========================================
# হেল্পার ফাংশন: জিওমেট্রি এবং ফিজিক্স
# ==========================================
def calculate_physics(period_days, depth, star_mass, star_radius):
    r_star = star_radius * u.R_sun
    m_star = star_mass * u.M_sun

    # Calculate Rp/R* (Ratio of planet radius to star radius)
    rp_rs_ratio = float(np.sqrt(depth))

    # Calculate Semi-major axis (distance)
    period = period_days * u.day
    a_cubed = (const.G * m_star * period**2) / (4 * np.pi**2)
    a_m = np.cbrt(a_cubed.to_value(u.m**3)) * u.m
    a_au = float(a_m.to_value(u.AU))
    
    return rp_rs_ratio, a_au

# ==========================================
# বেয়েসিয়ান মডেল (Pure JAX Native Model)
# ==========================================
def transit_mcmc_model(time, initial_period, initial_depth, flux=None):
    # প্রাইয়র (Priors) সেট করা
    period = numpyro.sample("period", dist.Normal(initial_period, 0.1))
    depth = numpyro.sample("depth", dist.Normal(initial_depth, 0.002))
    duration = numpyro.sample("duration", dist.Uniform(0.05, 0.2))
    t0 = numpyro.sample("t0", dist.Normal(0.0, 0.1))
    
    # ফেজ বা সময়কাল হিসাব
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    
    # Buggy exoplanet-core এর বদলে JAX-এর নিজস্ব গাণিতিক মডেল (NUTS-এর জন্য অত্যন্ত নিরাপদ)
    dip = depth * jnp.exp(-0.5 * (phase / (duration / 2))**2)
    
    # এক্সপেক্টেড লাইট কার্ভ
    mu = 1.0 - dip 
    
    # অবজারভেশন
    numpyro.sample("obs", dist.Normal(mu, 0.002), obs=flux)

# ==========================================
# STEP 9 & 10: The Master `fit()` Function
# ==========================================
def fit(time, flux, star_mass=1.0, star_radius=1.0):
    """
    সম্পূর্ণ পাইপলাইন (BLS -> MCMC -> Physics) একত্রে রান করে এবং 
    ফ্রন্টএন্ডের জন্য নির্দিষ্ট ফরম্যাটে ডেটা রিটার্ন করে।
    """
    print("-> Step 1: BLS ব্যবহার করে সিগন্যাল খোঁজা হচ্ছে...")
    model = BoxLeastSquares(time, flux)
    results = model.power(np.linspace(0.5, 20.0, 1000), np.linspace(0.01, 0.2, 50))
    bls_period = results.period[np.argmax(results.power)]
    bls_depth = results.depth[np.argmax(results.power)]
    
    print("-> Step 2: Pure JAX ও MCMC (NUTS) দিয়ে নিখুঁত মান বের করা হচ্ছে...")
    # NUTS স্যাম্পলার ঠিকভাবেই কাজ করবে কারণ মডেলটি পুরোপুরি JAX-এর সাহায্যে তৈরি
    kernel = NUTS(transit_mcmc_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=False)
    mcmc.run(random.PRNGKey(0), time=jnp.array(time), initial_period=bls_period, initial_depth=bls_depth, flux=jnp.array(flux))
    
    # MCMC থেকে গড় মান এবং র-স্যাম্পল বের করা
    samples = mcmc.get_samples()
    final_period = float(np.mean(samples['period']))
    final_depth = float(np.mean(samples['depth']))
    
    print("-> Step 3: ফিজিক্স এবং জিওমেট্রি হিসাব করা হচ্ছে...")
    rp_rs, a_au = calculate_physics(final_period, final_depth, star_mass, star_radius)
    # ==========================================
    # Standardized Dashboard Payload (For Member D)
    # ==========================================
    payload = {
        "period": round(final_period, 4),
        "Rp/R*": round(rp_rs, 4),
        "semi_major_axis": round(a_au, 4),
        "posterior_distributions": {
            "period": samples['period'].tolist(),
            "depth": samples['depth'].tolist(),
            "duration": samples['duration'].tolist(),
            "t0": samples['t0'].tolist()
        }
    }
    return payload

# ==========================================
# টেস্ট রান
# ==========================================
# --- ফাইলের একদম শেষে এই অংশটুকু বসবে ---

# --- ফাইলের একদম শেষে এই অংশটুকু বসবে ---

if __name__ == "__main__":
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import pandas as pd  # আসল ডেটা (CSV) লোড করার জন্য

    print("--- Pipeline Started ---")
    
    # ==========================================
    # পয়েন্ট ৪: আসল ডেটা ইনপুট দেওয়া 
    # ==========================================
    # (আপনার কাছে যদি আসল ডেটার CSV ফাইল থাকে, তবে নিচের ৩ লাইনের কমেন্ট তুলে দিন 
    # এবং "your_real_data.csv" এর জায়গায় আপনার ফাইলের নাম দিন)
    
    # df = pd.read_csv("your_real_data.csv")
    # time_subset = df['time'].values
    # flux_subset = df['flux'].values
    
    # আপাতত টেস্টিংয়ের জন্য ডেমো ডেটা রাখছি (যাতে কোড ক্র্যাশ না করে)
    time_subset = np.linspace(0, 20, 500)
    flux_subset = np.random.normal(1, 0.005, 500)

    # নক্ষত্রের ডেটা
    test_star_mass = 1.05
    test_star_radius = 1.06
    
    print("Running pipeline... Please wait.")
    real_results = fit(time_subset, flux_subset, star_mass=test_star_mass, star_radius=test_star_radius)
    
    # স্ক্রিন পরিষ্কার রাখার জন্য শুধু মূল রেজাল্ট প্রিন্ট করা হচ্ছে
    print("\n--- Final Results ---")
    print(f"Period: {real_results['period']} days")
    print(f"Radius Ratio (Rp/R*): {real_results['Rp/R*']}")
    print(f"Semi-major Axis: {real_results['semi_major_axis']} AU")

    # ==========================================
    # পয়েন্ট ২: ডেটা JSON ফাইলে সেভ করা
    # ==========================================
    print("\nSaving massive posterior data to JSON file...")
    
    # Numpy array-কে JSON-এ সেভ করা যায় না, তাই সেগুলোকে Normal List-এ কনভার্ট করতে হয়
    results_for_json = {
        "period": real_results["period"],
        "Rp/R*": real_results["Rp/R*"],
        "semi_major_axis": real_results["semi_major_axis"],
        "posterior_distributions": {
            key: np.array(val).tolist() for key, val in real_results["posterior_distributions"].items()
        }
    }

    # ডেটা সেভ করা হচ্ছে pipeline_results.json ফাইলে
    with open("pipeline_results.json", "w") as file:
        json.dump(results_for_json, file, indent=4)
    print("Success! Data saved to 'pipeline_results.json'.")

    # ==========================================
    # পয়েন্ট ৩: ডেটা ভিজ্যুয়ালাইজেশন (গ্রাফ তৈরি)
    # ==========================================
    print("\nGenerating Light Curve Plot...")
    plt.figure(figsize=(10, 5))
    plt.scatter(time_subset, flux_subset, s=5, color='black', alpha=0.7, label='Data (Flux)')
    
    # গ্রাফ সাজানো
    plt.xlabel("Time (Days)")
    plt.ylabel("Normalized Flux")
    plt.title("Exoplanet Transit Light Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # গ্রাফটি স্ক্রিনে দেখানো
    plt.show()