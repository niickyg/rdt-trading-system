"""
Paper trading broker for simulation and testing.

This module re-exports from brokers.paper_broker for backward compatibility.
New code should import from brokers.paper_broker or use brokers.get_broker("paper").
"""

# Re-export PaperBroker from the new location for backward compatibility
from brokers.paper_broker import PaperBroker

__all__ = ['PaperBroker']
