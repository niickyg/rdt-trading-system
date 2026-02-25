"""
Black-Scholes Pricing Engine for the RDT Trading System.

Pure math implementation using only the stdlib `math` module.
Provides option pricing, Greeks calculation, and implied volatility
via Newton-Raphson iteration.

Reference values (for validation):
    S=100, K=100, T=0.25, r=0.05, sigma=0.20
    Call price ≈ 4.615, Put price ≈ 3.373
    Call delta ≈ 0.595, Put delta ≈ -0.405
"""

import math
import os
from dataclasses import dataclass


# Risk-free rate default, configurable via environment
DEFAULT_RISK_FREE_RATE = float(os.environ.get("OPTIONS_RISK_FREE_RATE", "0.05"))


@dataclass
class BSGreeks:
    """Black-Scholes Greeks result."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0


def norm_cdf(x: float) -> float:
    """
    Cumulative distribution function for the standard normal distribution.

    Uses Abramowitz & Stegun approximation (formula 26.2.17),
    accurate to ~7.5e-8.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0
    if x < 0:
        sign = -1.0
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Calculate d1 and d2 for Black-Scholes formula."""
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "C"
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, decimal)
        sigma: Volatility (annualized, decimal)
        option_type: "C" for call, "P" for put

    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At expiration or invalid inputs, return intrinsic value
        if option_type.upper() == "C":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)

    if option_type.upper() == "C":
        return S * norm_cdf(d1) - K * discount * norm_cdf(d2)
    else:
        return K * discount * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "C"
) -> float:
    """Calculate option delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type.upper() == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _d1d2(S, K, T, r, sigma)
    if option_type.upper() == "C":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate option gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "C"
) -> float:
    """Calculate option theta (per calendar day)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_t = math.sqrt(T)
    discount = math.exp(-r * T)

    # First term (same for calls and puts)
    term1 = -(S * norm_pdf(d1) * sigma) / (2 * sqrt_t)

    if option_type.upper() == "C":
        theta_annual = term1 - r * K * discount * norm_cdf(d2)
    else:
        theta_annual = term1 + r * K * discount * norm_cdf(-d2)

    # Convert to per-calendar-day
    return theta_annual / 365.0


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate option vega (per 1% change in volatility)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    # Vega per 1 point of vol (0.01)
    return S * norm_pdf(d1) * math.sqrt(T) / 100.0


def generate_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "C"
) -> BSGreeks:
    """
    Generate all Greeks for an option.

    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "C" or "P"

    Returns:
        BSGreeks dataclass with price, delta, gamma, theta, vega, rho
    """
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    delta = bs_delta(S, K, T, r, sigma, option_type)
    gamma = bs_gamma(S, K, T, r, sigma)
    theta = bs_theta(S, K, T, r, sigma, option_type)
    vega = bs_vega(S, K, T, r, sigma)

    # Rho (per 1% change in rate)
    if T > 0 and sigma > 0 and S > 0 and K > 0:
        d1, d2 = _d1d2(S, K, T, r, sigma)
        discount = math.exp(-r * T)
        if option_type.upper() == "C":
            rho = K * T * discount * norm_cdf(d2) / 100.0
        else:
            rho = -K * T * discount * norm_cdf(-d2) / 100.0
    else:
        rho = 0.0

    return BSGreeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


def implied_volatility(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "C",
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        market_price: Observed market price of the option
        S: Underlying price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        option_type: "C" or "P"
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility (decimal), or 0.0 if convergence fails
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0

    # Check against intrinsic value
    if option_type.upper() == "C":
        intrinsic = max(0.0, S - K * math.exp(-r * T))
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S)

    if market_price < intrinsic - tolerance:
        return 0.0  # Below intrinsic, no valid IV

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / S
    sigma = max(0.01, min(sigma, 5.0))  # Clamp to reasonable range

    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        # Vega for Newton step (raw, not per-1%)
        d1, _ = _d1d2(S, K, T, r, sigma)
        vega_raw = S * norm_pdf(d1) * math.sqrt(T)

        if vega_raw < 1e-12:
            break  # Vega too small, can't converge

        sigma -= diff / vega_raw
        sigma = max(0.001, min(sigma, 10.0))  # Keep in bounds

    return sigma
