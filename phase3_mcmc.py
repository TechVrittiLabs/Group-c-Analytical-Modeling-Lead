import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np

# JAX-কে 64-bit precision ব্যবহার করতে বাধ্য করা (Phase 1-এর নিয়ম)
jax.config.update("jax_enable_x64", True)

# ==========================================
# STEP 7: Hook up the Light Curve Evaluator
# (JAX-এর সাহায্যে একটি ডিফারেন্সিয়েবল মডেল তৈরি)
# ==========================================
def jax_transit_model(time, period, depth, duration, t0):
    """
    একটি মসৃণ (smooth) ট্রানজিট মডেল যা JAX সহজে ক্যালকুলেট করতে পারে।
    আমরা এখানে একটি inverted Gaussian ব্যবহার করছি যাতে গ্রেডিয়েন্ট (gradients) বের করা সহজ হয়।
    """
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    # আলো কমার (dip) পরিমাণ হিসাব করা
    dip = depth * jnp.exp(-0.5 * (phase / (duration / 2))**2)
    return 1.0 - dip

# ==========================================
# STEP 6: Define the NumPyro Probabilistic Model
# ==========================================
def transit_mcmc_model(time, flux=None):
    """
    বেয়েসিয়ান প্রাইয়র (Priors) এবং লাইকলিহুড (Likelihood) সেটআপ।
    """
    # আমরা BLS থেকে পাওয়া ধারণা অনুযায়ী কিছু প্রাইয়র সেট করছি
    period = numpyro.sample("period", dist.TruncatedNormal(loc=4.05, scale=0.1, low=3.0, high=5.0))
    depth = numpyro.sample("depth", dist.Uniform(0.005, 0.015))
    duration = numpyro.sample("duration", dist.Uniform(0.05, 0.2))
    t0 = numpyro.sample("t0", dist.Normal(0.0, 0.1))
    
    # মডেল থেকে লাইট কার্ভ তৈরি
    mu = jax_transit_model(time, period, depth, duration, t0)
    
    # STEP 8: লাইকলিহুড - আমাদের মডেলের সাথে আসল ডেটার তুলনা
    sigma = 0.002 # নয়েজ লেভেল
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=flux)

# ==========================================
# Phase 3 Execution & NUTS Sampler
# ==========================================
def run_phase3():
    print("১. ফেক ডেটা তৈরি করা হচ্ছে...")
    time = np.linspace(0, 20, 20 * 24)
    true_period, true_depth, true_duration, true_t0 = 4.05, 0.009, 0.1, 0.0
    
    # ফেক সিগন্যাল তৈরি
    phase = (time - true_t0 + 0.5 * true_period) % true_period - 0.5 * true_period
    flux = 1.0 - true_depth * np.exp(-0.5 * (phase / (true_duration / 2))**2)
    flux += np.random.normal(0, 0.002, size=len(time)) # নয়েজ যোগ করা হলো
    
    print("২. NumPyro NUTS স্যাম্পলার চালু করা হচ্ছে (এটি একটু সময় নিতে পারে)...")
    
    # MCMC ইঞ্জিন সেটআপ (num_warmup এবং num_samples দিয়ে স্পিড কন্ট্রোল করা হয়)
    rng_key = random.PRNGKey(0)
    kernel = NUTS(transit_mcmc_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=True)
    
    # ডেটা দিয়ে মডেলটি রান করানো
    mcmc.run(rng_key, time=jnp.array(time), flux=jnp.array(flux))
    
    print("\n৩. MCMC চেইন সম্পন্ন হয়েছে! চূড়ান্ত ফলাফল (Posteriors):")
    mcmc.print_summary()

if __name__ == "__main__":
    run_phase3()