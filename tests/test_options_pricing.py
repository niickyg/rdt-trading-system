"""
Tests for options/pricing.py — Black-Scholes pricing engine.

Validates pricing, Greeks, and implied volatility against known values.
"""

import math
import pytest

from options.pricing import (
    norm_cdf, norm_pdf, black_scholes_price, bs_delta, bs_gamma,
    bs_theta, bs_vega, generate_greeks, implied_volatility,
    BSGreeks,
)


# ---------------------------------------------------------------------------
# Known reference values
# S=100, K=100, T=0.25 (3 months), r=0.05, sigma=0.20
# Expected: Call ≈ 5.88, Put ≈ 4.64
# ---------------------------------------------------------------------------

S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20


class TestNormCDF:
    def test_at_zero(self):
        assert abs(norm_cdf(0.0) - 0.5) < 1e-6

    def test_at_large_positive(self):
        assert abs(norm_cdf(6.0) - 1.0) < 1e-6

    def test_at_large_negative(self):
        assert abs(norm_cdf(-6.0) - 0.0) < 1e-6

    def test_symmetry(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(norm_cdf(x) + norm_cdf(-x) - 1.0) < 1e-7

    def test_known_value(self):
        # N(1.0) ≈ 0.8413
        assert abs(norm_cdf(1.0) - 0.8413) < 0.001


class TestNormPDF:
    def test_at_zero(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(norm_pdf(0.0) - expected) < 1e-10

    def test_symmetry(self):
        assert abs(norm_pdf(1.5) - norm_pdf(-1.5)) < 1e-10


class TestBlackScholesPrice:
    def test_call_price_known_value(self):
        price = black_scholes_price(S, K, T, r, sigma, "C")
        assert abs(price - 4.615) < 0.01

    def test_put_price_known_value(self):
        price = black_scholes_price(S, K, T, r, sigma, "P")
        assert abs(price - 3.373) < 0.01

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        call = black_scholes_price(S, K, T, r, sigma, "C")
        put = black_scholes_price(S, K, T, r, sigma, "P")
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 1e-6

    def test_deep_itm_call(self):
        # Deep ITM call should approach intrinsic value
        price = black_scholes_price(200, 100, 0.25, 0.05, 0.20, "C")
        assert price > 99  # At least intrinsic (100)

    def test_deep_otm_call(self):
        # Deep OTM call should be near zero
        price = black_scholes_price(50, 100, 0.25, 0.05, 0.20, "C")
        assert price < 0.01

    def test_at_expiration(self):
        assert black_scholes_price(110, 100, 0.0, 0.05, 0.20, "C") == 10.0
        assert black_scholes_price(90, 100, 0.0, 0.05, 0.20, "C") == 0.0
        assert black_scholes_price(90, 100, 0.0, 0.05, 0.20, "P") == 10.0
        assert black_scholes_price(110, 100, 0.0, 0.05, 0.20, "P") == 0.0

    def test_zero_vol(self):
        # With zero vol, option = intrinsic discounted
        call = black_scholes_price(110, 100, 0.25, 0.05, 0.0, "C")
        assert call == max(0, 110 - 100)

    def test_negative_inputs(self):
        assert black_scholes_price(-1, 100, 0.25, 0.05, 0.20, "C") == 0.0


class TestDelta:
    def test_call_delta_atm(self):
        delta = bs_delta(S, K, T, r, sigma, "C")
        # ATM call delta ≈ 0.55-0.60
        assert 0.50 < delta < 0.70

    def test_put_delta_atm(self):
        delta = bs_delta(S, K, T, r, sigma, "P")
        # ATM put delta ≈ -0.45 to -0.40
        assert -0.50 < delta < -0.30

    def test_call_put_delta_relationship(self):
        """Delta_call - Delta_put = 1"""
        call_d = bs_delta(S, K, T, r, sigma, "C")
        put_d = bs_delta(S, K, T, r, sigma, "P")
        assert abs((call_d - put_d) - 1.0) < 1e-6

    def test_deep_itm_call_delta(self):
        delta = bs_delta(200, 100, 0.25, 0.05, 0.20, "C")
        assert delta > 0.99

    def test_deep_otm_call_delta(self):
        delta = bs_delta(50, 100, 0.25, 0.05, 0.20, "C")
        assert delta < 0.01

    def test_at_expiration(self):
        assert bs_delta(110, 100, 0.0, 0.05, 0.20, "C") == 1.0
        assert bs_delta(90, 100, 0.0, 0.05, 0.20, "C") == 0.0


class TestGamma:
    def test_gamma_positive(self):
        gamma = bs_gamma(S, K, T, r, sigma)
        assert gamma > 0

    def test_gamma_highest_atm(self):
        gamma_atm = bs_gamma(100, 100, T, r, sigma)
        gamma_otm = bs_gamma(100, 120, T, r, sigma)
        gamma_itm = bs_gamma(100, 80, T, r, sigma)
        assert gamma_atm > gamma_otm
        assert gamma_atm > gamma_itm

    def test_gamma_at_expiration(self):
        assert bs_gamma(100, 100, 0.0, 0.05, 0.20) == 0.0


class TestTheta:
    def test_call_theta_negative(self):
        theta = bs_theta(S, K, T, r, sigma, "C")
        assert theta < 0  # Time decay is negative

    def test_put_theta_usually_negative(self):
        theta = bs_theta(S, K, T, r, sigma, "P")
        # ATM put theta is typically negative
        assert theta < 0


class TestVega:
    def test_vega_positive(self):
        vega = bs_vega(S, K, T, r, sigma)
        assert vega > 0

    def test_vega_highest_atm(self):
        vega_atm = bs_vega(100, 100, T, r, sigma)
        vega_otm = bs_vega(100, 120, T, r, sigma)
        assert vega_atm > vega_otm


class TestGenerateGreeks:
    def test_returns_bsgreeks(self):
        result = generate_greeks(S, K, T, r, sigma, "C")
        assert isinstance(result, BSGreeks)

    def test_all_fields_populated(self):
        g = generate_greeks(S, K, T, r, sigma, "C")
        assert g.price > 0
        assert 0 < g.delta < 1
        assert g.gamma > 0
        assert g.theta < 0
        assert g.vega > 0
        assert g.rho > 0  # Call rho is positive

    def test_put_rho_negative(self):
        g = generate_greeks(S, K, T, r, sigma, "P")
        assert g.rho < 0


class TestImpliedVolatility:
    def test_round_trip_call(self):
        """Price with known vol, then recover vol from price."""
        price = black_scholes_price(S, K, T, r, sigma, "C")
        iv = implied_volatility(price, S, K, T, r, "C")
        assert abs(iv - sigma) < 0.001

    def test_round_trip_put(self):
        price = black_scholes_price(S, K, T, r, sigma, "P")
        iv = implied_volatility(price, S, K, T, r, "P")
        assert abs(iv - sigma) < 0.001

    def test_high_vol(self):
        high_sigma = 0.80
        price = black_scholes_price(S, K, T, r, high_sigma, "C")
        iv = implied_volatility(price, S, K, T, r, "C")
        assert abs(iv - high_sigma) < 0.001

    def test_low_vol(self):
        low_sigma = 0.05
        price = black_scholes_price(S, K, T, r, low_sigma, "C")
        iv = implied_volatility(price, S, K, T, r, "C")
        assert abs(iv - low_sigma) < 0.005

    def test_zero_price(self):
        iv = implied_volatility(0.0, S, K, T, r, "C")
        assert iv == 0.0

    def test_below_intrinsic(self):
        iv = implied_volatility(0.01, 110, 100, T, r, "C")
        # Price below intrinsic (~10), should return 0
        assert iv == 0.0

    def test_otm_option(self):
        price = black_scholes_price(100, 110, 0.25, 0.05, 0.30, "C")
        iv = implied_volatility(price, 100, 110, 0.25, 0.05, "C")
        assert abs(iv - 0.30) < 0.005

    def test_various_maturities(self):
        for T_val in [0.01, 0.1, 0.5, 1.0, 2.0]:
            price = black_scholes_price(S, K, T_val, r, sigma, "C")
            iv = implied_volatility(price, S, K, T_val, r, "C")
            assert abs(iv - sigma) < 0.01, f"Failed for T={T_val}"
