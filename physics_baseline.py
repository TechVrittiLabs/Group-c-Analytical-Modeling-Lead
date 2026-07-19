import numpy as np
import jax
import astropy.units as u
import astropy.constants as const

# ==========================================
# STEP 1: Set up the Environment
# ==========================================

# Enforce 64-bit precision in JAX to prevent silent truncation errors
# during later Bayesian inference steps.
jax.config.update("jax_enable_x64", True)

# ==========================================
# STEP 2: Implement the Geometry Module
# ==========================================

def calculate_planet_radius(transit_depth, star_radius):
    """
    Calculates the planet's radius in Earth radii based on transit depth.
    
    Args:
        transit_depth (float): The fractional drop in flux (e.g., 0.009025).
        star_radius (astropy.units.Quantity): The radius of the host star.
        
    Returns:
        astropy.units.Quantity: Planet radius in Earth radii.
    """
    # The depth is the square of the radius ratio: R_p / R_* = sqrt(depth)
    radius_ratio = np.sqrt(transit_depth)
    
    # Calculate physical radius and convert to Earth Radii
    planet_radius = (radius_ratio * star_radius).to(u.R_earth)
    return planet_radius

def calculate_semi_major_axis(orbital_period, star_mass):
    """
    Calculates the semi-major axis (a) using Kepler's Third Law.
    
    Args:
        orbital_period (astropy.units.Quantity): The orbital period of the planet.
        star_mass (astropy.units.Quantity): The mass of the host star.
        
    Returns:
        astropy.units.Quantity: Semi-major axis in Astronomical Units (AU).
    """
    # Kepler's Third Law: a^3 = (G * M * P^2) / (4 * pi^2)
    numerator = const.G * star_mass * orbital_period**2
    denominator = 4 * np.pi**2
    a_cubed = numerator / denominator
    
    # Take the cube root and convert to AU
    # We convert to a base unit (m^3) first to safely apply the cube root
    a = np.cbrt(a_cubed.to(u.m**3).value) * u.m 
    return a.to(u.AU)

# ==========================================
# STEP 3: Establish the Task 17 Unit Test
# ==========================================

def test_task17_baseline():
    """
    Unit test to guarantee deterministic physics calculations match known 
    values before moving to probabilistic modeling.
    """
    # Known inputs for a Jupiter-sized planet around a Sun-like star
    # Note: Corrected the physical depth to 0.009025 (a ~0.9% flux drop) 
    # rather than 10.0, which would be physically impossible.
    depth = 0.009025 
    period = 4.05 * u.day
    r_star = 1.0 * u.R_sun
    m_star = 1.0 * u.M_sun
    
    # Run the module functions
    rp_calculated = calculate_planet_radius(depth, r_star)
    a_calculated = calculate_semi_major_axis(period, m_star)
    
    # Expected baseline outputs
    expected_rp = 10.36 # Earth radii
    expected_a = 0.0497 # AU
    
    # Use np.isclose to account for minor floating-point differences 
    # or updates in astropy constant definitions.
    assert np.isclose(rp_calculated.value, expected_rp, atol=0.01), \
        f"Radius mismatch: Expected {expected_rp}, got {rp_calculated.value:.2f}"
        
    assert np.isclose(a_calculated.value, expected_a, atol=0.0001), \
        f"Semi-major axis mismatch: Expected {expected_a}, got {a_calculated.value:.4f}"
        
    print("✅ Task 17 Unit Test Passed: Standalone Physics & Geometry are locked in.")

# Run the test when the script is executed
if __name__ == "__main__":
    test_task17_baseline()