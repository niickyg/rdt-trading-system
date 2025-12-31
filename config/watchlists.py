"""
Curated Watchlists for RDT Trading System
Organized by liquidity, sector, and trading characteristics
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class WatchlistMetadata:
    """Metadata for a watchlist"""
    name: str
    description: str
    avg_volume_requirement: int
    sector_focus: str
    count: int


# =============================================================================
# CORE WATCHLISTS - High Liquidity S&P 500 Components
# =============================================================================

# Top 50 most liquid stocks - primary trading universe
SP500_TOP_50 = [
    # Mega-cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Large-cap Tech
    'AVGO', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO', 'ORCL', 'QCOM',
    'TXN', 'AMAT', 'MU', 'INTU', 'NOW', 'IBM', 'ADI', 'LRCX',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    # Consumer
    'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'WMT', 'PG', 'KO'
]

# Extended S&P 500 components with good liquidity
SP500_EXTENDED = [
    # Industrial
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'UNP',
    'FDX', 'CSX', 'NSC', 'WM', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
    'HAL', 'DVN', 'HES', 'FANG', 'BKR',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG',
    # Utilities & REITs
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC',
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O', 'WELL', 'DLR', 'AVB',
    # Consumer Staples
    'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'STZ',
    # Healthcare Extended
    'GILD', 'AMGN', 'ISRG', 'REGN', 'VRTX', 'BIIB', 'MDT', 'SYK', 'BDX', 'ZTS',
    # Tech Extended
    'PYPL', 'NFLX', 'SHOP', 'SQ', 'UBER', 'ABNB', 'DDOG', 'SNOW', 'ZS', 'CRWD',
    # Telecom & Media
    'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'WBD',
    # Financials Extended
    'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'AIG', 'MET', 'PRU', 'ALL', 'TRV',
    # Consumer Discretionary
    'AMZN', 'BKNG', 'LULU', 'RCL', 'CCL', 'MAR', 'HLT', 'YUM', 'DPZ', 'CMG'
]


# =============================================================================
# SECTOR-SPECIFIC WATCHLISTS
# =============================================================================

SECTOR_TECHNOLOGY = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'ADBE', 'CRM',
    'AMD', 'INTC', 'CSCO', 'ORCL', 'QCOM', 'TXN', 'AMAT', 'MU',
    'INTU', 'NOW', 'IBM', 'ADI', 'LRCX', 'KLAC', 'MCHP', 'SNPS',
    'CDNS', 'FTNT', 'PANW', 'NFLX', 'PYPL', 'SHOP'
]

SECTOR_FINANCIALS = [
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'SCHW',
    'V', 'MA', 'AXP', 'COF', 'DFS', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO',
    'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'AFL', 'PGR', 'CB', 'MMC', 'AON'
]

SECTOR_HEALTHCARE = [
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'MDT', 'SYK', 'ISRG', 'BDX',
    'ELV', 'CI', 'HUM', 'CVS', 'MCK', 'CAH', 'ABC', 'ZTS', 'DXCM', 'IDXX'
]

SECTOR_ENERGY = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
    'HAL', 'DVN', 'HES', 'FANG', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG'
]

SECTOR_CONSUMER = [
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'WMT',
    'PG', 'KO', 'PEP', 'BKNG', 'LULU', 'RCL', 'CCL', 'MAR', 'HLT', 'YUM',
    'DPZ', 'CMG', 'DG', 'DLTR', 'ROST', 'TJX', 'BBY', 'ORLY', 'AZO', 'ULTA'
]

SECTOR_INDUSTRIALS = [
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'UNP',
    'FDX', 'CSX', 'NSC', 'WM', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI',
    'NOC', 'GD', 'TDG', 'CARR', 'OTIS', 'JCI', 'SWK', 'FAST', 'PCAR', 'WAB'
]


# =============================================================================
# ETF WATCHLISTS - For sector rotation and market breadth
# =============================================================================

SECTOR_ETFS = [
    'XLK',  # Technology
    'XLF',  # Financials
    'XLV',  # Healthcare
    'XLE',  # Energy
    'XLY',  # Consumer Discretionary
    'XLP',  # Consumer Staples
    'XLI',  # Industrials
    'XLB',  # Materials
    'XLU',  # Utilities
    'XLRE', # Real Estate
    'XLC',  # Communication Services
]

MAJOR_ETFS = [
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq 100
    'IWM',  # Russell 2000
    'DIA',  # Dow Jones
    'MDY',  # Mid-Cap
    'VTI',  # Total Market
    'EFA',  # Developed Markets
    'EEM',  # Emerging Markets
]

LEVERAGED_ETFS = [
    'TQQQ', # 3x Nasdaq
    'SQQQ', # -3x Nasdaq
    'UPRO', # 3x S&P
    'SPXU', # -3x S&P
    'SOXL', # 3x Semiconductors
    'SOXS', # -3x Semiconductors
    'TNA',  # 3x Russell 2000
    'TZA',  # -3x Russell 2000
    'LABU', # 3x Biotech
    'LABD', # -3x Biotech
    'FAS',  # 3x Financials
    'FAZ',  # -3x Financials
]


# =============================================================================
# SPECIALTY WATCHLISTS
# =============================================================================

# High volatility stocks for aggressive trading
HIGH_VOLATILITY = [
    'TSLA', 'NVDA', 'AMD', 'COIN', 'SHOP', 'SQ', 'ROKU', 'SNAP',
    'RBLX', 'PLTR', 'HOOD', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    'MRNA', 'BNTX', 'NKLA', 'SPCE', 'GME', 'AMC'
]

# Dividend aristocrats - more stable for swing trades
DIVIDEND_ARISTOCRATS = [
    'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'ABT', 'ABBV', 'WMT', 'XOM', 'CVX',
    'CL', 'EMR', 'SWK', 'GPC', 'BDX', 'AFL', 'SHW', 'ADP', 'ITW', 'ED',
    'CLX', 'PPG', 'CINF', 'KMB', 'CTAS', 'LEG', 'TGT', 'MCD', 'LOW', 'CAT'
]

# International ADRs with good US liquidity
INTERNATIONAL_ADRS = [
    'TSM', 'BABA', 'NVO', 'ASML', 'TM', 'SAP', 'SHEL', 'UL', 'BP', 'HSBC',
    'SNY', 'GSK', 'AZN', 'RIO', 'BHP', 'VALE', 'PBR', 'SAN', 'INFY', 'HDB'
]


# =============================================================================
# COMBINED WATCHLISTS
# =============================================================================

def get_full_watchlist() -> List[str]:
    """Get comprehensive watchlist (150+ stocks)"""
    all_symbols = set()
    all_symbols.update(SP500_TOP_50)
    all_symbols.update(SP500_EXTENDED)
    # Remove duplicates and return sorted
    return sorted(list(all_symbols))


def get_core_watchlist() -> List[str]:
    """Get core watchlist (50 stocks) - most liquid"""
    return SP500_TOP_50.copy()


def get_aggressive_watchlist() -> List[str]:
    """Get aggressive watchlist including high volatility and leveraged ETFs"""
    watchlist = set(SP500_TOP_50)
    watchlist.update(HIGH_VOLATILITY)
    watchlist.update(LEVERAGED_ETFS)  # Include leveraged ETFs for amplified returns
    return sorted(list(watchlist))


def get_sector_watchlist(sector: str) -> List[str]:
    """Get sector-specific watchlist"""
    sector_map = {
        'technology': SECTOR_TECHNOLOGY,
        'tech': SECTOR_TECHNOLOGY,
        'financials': SECTOR_FINANCIALS,
        'finance': SECTOR_FINANCIALS,
        'healthcare': SECTOR_HEALTHCARE,
        'health': SECTOR_HEALTHCARE,
        'energy': SECTOR_ENERGY,
        'consumer': SECTOR_CONSUMER,
        'industrials': SECTOR_INDUSTRIALS,
        'industrial': SECTOR_INDUSTRIALS,
    }
    return sector_map.get(sector.lower(), SP500_TOP_50).copy()


def get_etf_watchlist() -> List[str]:
    """Get ETF-only watchlist"""
    etfs = set()
    etfs.update(SECTOR_ETFS)
    etfs.update(MAJOR_ETFS)
    return sorted(list(etfs))


def get_watchlist_by_name(name: str) -> List[str]:
    """Get watchlist by name"""
    watchlist_map = {
        'core': get_core_watchlist,
        'full': get_full_watchlist,
        'aggressive': get_aggressive_watchlist,
        'etfs': get_etf_watchlist,
        'dividends': lambda: DIVIDEND_ARISTOCRATS.copy(),
        'international': lambda: INTERNATIONAL_ADRS.copy(),
        'high_volatility': lambda: HIGH_VOLATILITY.copy(),
    }

    getter = watchlist_map.get(name.lower())
    if getter:
        return getter()

    # Try sector
    return get_sector_watchlist(name)


def get_all_watchlists_metadata() -> Dict[str, WatchlistMetadata]:
    """Get metadata for all available watchlists"""
    return {
        'core': WatchlistMetadata(
            name='Core',
            description='Top 50 most liquid S&P 500 stocks',
            avg_volume_requirement=5_000_000,
            sector_focus='Diversified',
            count=len(SP500_TOP_50)
        ),
        'full': WatchlistMetadata(
            name='Full',
            description='Comprehensive 150+ stock universe',
            avg_volume_requirement=1_000_000,
            sector_focus='Diversified',
            count=len(get_full_watchlist())
        ),
        'technology': WatchlistMetadata(
            name='Technology',
            description='Technology sector stocks',
            avg_volume_requirement=2_000_000,
            sector_focus='Technology',
            count=len(SECTOR_TECHNOLOGY)
        ),
        'financials': WatchlistMetadata(
            name='Financials',
            description='Financial sector stocks',
            avg_volume_requirement=2_000_000,
            sector_focus='Financials',
            count=len(SECTOR_FINANCIALS)
        ),
        'healthcare': WatchlistMetadata(
            name='Healthcare',
            description='Healthcare sector stocks',
            avg_volume_requirement=2_000_000,
            sector_focus='Healthcare',
            count=len(SECTOR_HEALTHCARE)
        ),
        'energy': WatchlistMetadata(
            name='Energy',
            description='Energy sector stocks',
            avg_volume_requirement=2_000_000,
            sector_focus='Energy',
            count=len(SECTOR_ENERGY)
        ),
        'etfs': WatchlistMetadata(
            name='ETFs',
            description='Sector and market ETFs',
            avg_volume_requirement=10_000_000,
            sector_focus='ETFs',
            count=len(get_etf_watchlist())
        ),
        'aggressive': WatchlistMetadata(
            name='Aggressive',
            description='High volatility momentum stocks',
            avg_volume_requirement=1_000_000,
            sector_focus='High Volatility',
            count=len(get_aggressive_watchlist())
        ),
    }


# =============================================================================
# WATCHLIST BUILDER
# =============================================================================

class WatchlistBuilder:
    """Build custom watchlists with filtering"""

    def __init__(self):
        self.symbols: List[str] = []

    def add_core(self) -> 'WatchlistBuilder':
        """Add core S&P 500 stocks"""
        self.symbols.extend(SP500_TOP_50)
        return self

    def add_extended(self) -> 'WatchlistBuilder':
        """Add extended S&P 500 stocks"""
        self.symbols.extend(SP500_EXTENDED)
        return self

    def add_sector(self, sector: str) -> 'WatchlistBuilder':
        """Add sector-specific stocks"""
        self.symbols.extend(get_sector_watchlist(sector))
        return self

    def add_etfs(self) -> 'WatchlistBuilder':
        """Add ETFs"""
        self.symbols.extend(get_etf_watchlist())
        return self

    def add_high_volatility(self) -> 'WatchlistBuilder':
        """Add high volatility stocks"""
        self.symbols.extend(HIGH_VOLATILITY)
        return self

    def add_custom(self, symbols: List[str]) -> 'WatchlistBuilder':
        """Add custom symbols"""
        self.symbols.extend(symbols)
        return self

    def exclude(self, symbols: List[str]) -> 'WatchlistBuilder':
        """Exclude specific symbols"""
        exclude_set = set(s.upper() for s in symbols)
        self.symbols = [s for s in self.symbols if s.upper() not in exclude_set]
        return self

    def build(self) -> List[str]:
        """Build final deduplicated watchlist"""
        seen = set()
        result = []
        for symbol in self.symbols:
            symbol_upper = symbol.upper()
            if symbol_upper not in seen:
                seen.add(symbol_upper)
                result.append(symbol_upper)
        return sorted(result)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def print_watchlist_info():
    """Print information about available watchlists"""
    print("\n" + "=" * 60)
    print("AVAILABLE WATCHLISTS")
    print("=" * 60)

    for name, meta in get_all_watchlists_metadata().items():
        print(f"\n{meta.name} ({name}):")
        print(f"  Description: {meta.description}")
        print(f"  Stocks: {meta.count}")
        print(f"  Sector Focus: {meta.sector_focus}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_watchlist_info()

    print("\nFull watchlist count:", len(get_full_watchlist()))
    print("Core watchlist count:", len(get_core_watchlist()))

    # Example custom watchlist
    custom = (WatchlistBuilder()
              .add_core()
              .add_sector('technology')
              .add_etfs()
              .exclude(['GME', 'AMC'])
              .build())
    print(f"Custom watchlist count: {len(custom)}")
