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
def calculate_physics(period_days, depth):
    r_star = 1.0 * u.R_sun
    m_star = 1.0 * u.M_sun
    
    # Rp/R* (নক্ষত্র ও গ্রহের ব্যাসার্ধের অনুপাত) নির্ণয়
    rp_rs_ratio = float(np.sqrt(depth))
    
    # দূরত্ব (Semi-major axis) নির্ণয়
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
def fit(time, flux):
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
    rp_rs, a_au = calculate_physics(final_period, final_depth)
    
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
if __name__ == "__main__":
    print("ফেক ডেটা তৈরি করে আপডেট হওয়া পাইপলাইন টেস্ট করা হচ্ছে...\n")
    time_mock = np.linspace(0, 20, 20 * 24)
    phase = (time_mock - 0.0 + 0.5 * 4.05) % 4.05 - 0.5 * 4.05
    flux_mock = 1.0 - 0.009 * np.exp(-0.5 * (phase / (0.1 / 2))**2)
    flux_mock += np.random.normal(0, 0.002, size=len(time_mock))
    
    final_result = fit(time_mock, flux_mock)
    
    print("\n✅ চূড়ান্ত আউটপুট (ড্যাশবোর্ডে পাঠানোর জন্য প্রস্তুত):")
    display_result = {k: v for k, v in final_result.items() if k != 'posterior_distributions'}
    display_result['posterior_distributions'] = "<500 MCMC samples ready for frontend>"
    print(display_result)