"""
Integration Tests for RDT Trading System

These tests verify end-to-end functionality across multiple components:
- Trading flow (signal to trade execution)
- API authentication and endpoints
- WebSocket real-time communication
- ML pipeline (feature engineering, predictions, retraining)
- Alert system (multi-channel delivery)

Run integration tests with:
    pytest tests/integration/ -v -m integration

Run slow integration tests:
    pytest tests/integration/ -v -m "integration and slow"
"""
