"""
GraphQL API for RDT Trading System

Provides a flexible GraphQL endpoint for querying trading data including:
- Signals with filtering
- Positions and trades
- Portfolio summary
- Market and scanner status

Usage:
    from api.graphql import graphql_bp, schema
    app.register_blueprint(graphql_bp)
"""

from api.graphql.schema import schema
from api.graphql.views import graphql_bp
from api.graphql.auth import GraphQLAuthMiddleware

__all__ = ['schema', 'graphql_bp', 'GraphQLAuthMiddleware']
