"""
GraphQL Views and Flask Blueprint

Provides the /graphql endpoint with:
- POST for query execution
- GET for GraphQL Playground/GraphiQL
"""

import json
from flask import Blueprint, request, jsonify, render_template_string
from graphql import graphql_sync
from loguru import logger

from api.graphql.schema import schema
from api.graphql.auth import get_graphql_context


# Create blueprint
graphql_bp = Blueprint('graphql', __name__, url_prefix='/graphql')


@graphql_bp.before_request
def require_graphql_auth():
    from flask import request, jsonify
    api_key = request.headers.get('X-API-Key')
    if api_key:
        from api.v1.auth import api_key_manager
        is_valid, error = api_key_manager.validate_api_key(api_key)
        if is_valid:
            return None
    # Check Flask-Login session
    try:
        from flask_login import current_user
        if current_user and current_user.is_authenticated:
            return None
    except ImportError:
        pass
    return jsonify({'error': 'Authentication required'}), 401


# GraphiQL HTML template
GRAPHIQL_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>RDT Trading System - GraphQL Playground</title>
    <link href="https://unpkg.com/graphiql@2.4.7/graphiql.min.css" rel="stylesheet" />
    <style>
        body {
            height: 100vh;
            margin: 0;
            overflow: hidden;
        }
        #graphiql {
            height: 100vh;
        }
        .graphiql-container {
            font-family: system-ui, -apple-system, sans-serif;
        }
        .title-bar {
            background: #1a1a2e;
            color: white;
            padding: 10px 20px;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .title-bar h1 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }
        .title-bar a {
            color: #61dafb;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="title-bar">
        <h1>RDT Trading System - GraphQL API</h1>
        <div>
            <a href="/api/v1/docs" target="_blank">REST API Docs</a> |
            <a href="/" target="_blank">Dashboard</a>
        </div>
    </div>
    <div id="graphiql"></div>

    <script
        crossorigin
        src="https://unpkg.com/react@18/umd/react.production.min.js"
    ></script>
    <script
        crossorigin
        src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"
    ></script>
    <script
        crossorigin
        src="https://unpkg.com/graphiql@2.4.7/graphiql.min.js"
    ></script>

    <script>
        const apiKey = localStorage.getItem('rdt_api_key') || 'rdt_test_key_for_development_only';

        const fetcher = GraphiQL.createFetcher({
            url: '/graphql',
            headers: {
                'X-API-Key': apiKey,
            },
        });

        const defaultQuery = `# Welcome to RDT Trading System GraphQL API!
#
# Add your API key to headers or use the default test key.
# Click the "Headers" tab below and add:
# { "X-API-Key": "your_api_key_here" }

# Example queries:

# Get recent signals
query GetSignals {
  signals(limit: 10, direction: LONG) {
    id
    symbol
    direction
    strength
    rrs
    price
    timestamp
  }
}

# Get open positions
query GetPositions {
  positions {
    symbol
    direction
    entryPrice
    shares
    currentPrice
    pnl
    pnlPct
  }
}

# Get portfolio summary
query GetPortfolio {
  portfolio {
    totalValue
    dayPnl
    dayPnlPercent
    positionsCount
    performance {
      winRate
      profitFactor
      totalReturnPct
    }
  }
}

# Get market status
query MarketStatus {
  marketStatus {
    isOpen
    status
    currentTime
    timezone
  }
}

# Get scanner status
query ScannerStatus {
  scannerStatus {
    isRunning
    lastScan
    symbolsMonitored
    activeSignals
  }
}

# Get current user
query Me {
  me {
    id
    email
    tier
    rateLimit
    features
  }
}
`;

        ReactDOM.createRoot(document.getElementById('graphiql')).render(
            React.createElement(GraphiQL, {
                fetcher: fetcher,
                defaultQuery: defaultQuery,
                defaultVariableEditorOpen: false,
                headerEditorEnabled: true,
                shouldPersistHeaders: true,
            })
        );
    </script>
</body>
</html>
'''


@graphql_bp.route('', methods=['GET'])
@graphql_bp.route('/', methods=['GET'])
def graphql_playground():
    """
    Serve GraphQL Playground interface.

    This provides an interactive UI for exploring and testing the GraphQL API.
    """
    return render_template_string(GRAPHIQL_TEMPLATE)


@graphql_bp.route('', methods=['POST'])
@graphql_bp.route('/', methods=['POST'])
def graphql_endpoint():
    """
    Handle GraphQL query execution.

    Accepts JSON body with:
    - query: GraphQL query string
    - variables: Optional variables dict
    - operationName: Optional operation name

    Returns JSON response with:
    - data: Query results
    - errors: Any errors that occurred
    """
    # Get request data
    content_type = request.content_type or ''

    if 'application/json' in content_type:
        data = request.get_json()
    elif 'application/graphql' in content_type:
        data = {'query': request.data.decode('utf-8')}
    else:
        # Try to parse as JSON anyway
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({
                'errors': [{
                    'message': 'Invalid request format. Use application/json content type.'
                }]
            }), 400

    if not data:
        return jsonify({
            'errors': [{
                'message': 'No data provided'
            }]
        }), 400

    # Extract query parts
    query = data.get('query')
    variables = data.get('variables')
    operation_name = data.get('operationName')

    if not query:
        return jsonify({
            'errors': [{
                'message': 'No query provided'
            }]
        }), 400

    # Get authentication context
    context = get_graphql_context()

    # Log the request (without sensitive data)
    logger.debug(
        f"GraphQL request: operation={operation_name}, "
        f"authenticated={context.get('authenticated')}, "
        f"tier={context.get('tier')}"
    )

    try:
        # Execute the query
        result = graphql_sync(
            schema.graphql_schema,
            query,
            variable_values=variables,
            operation_name=operation_name,
            context_value=context,
        )

        # Build response
        response_data = {}

        if result.data:
            response_data['data'] = result.data

        if result.errors:
            response_data['errors'] = [
                {
                    'message': str(error.message),
                    'locations': [
                        {'line': loc.line, 'column': loc.column}
                        for loc in (error.locations or [])
                    ] if error.locations else None,
                    'path': error.path,
                }
                for error in result.errors
            ]

        # Log errors
        if result.errors:
            for error in result.errors:
                logger.warning(f"GraphQL error: {error.message}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"GraphQL execution error: {e}")
        return jsonify({
            'errors': [{
                'message': 'Internal server error'
            }]
        }), 500


@graphql_bp.route('/schema', methods=['GET'])
def get_schema():
    """
    Return the GraphQL schema in SDL format.

    Useful for code generation and documentation tools.
    """
    from graphql import print_schema
    sdl = print_schema(schema.graphql_schema)
    return sdl, 200, {'Content-Type': 'text/plain'}


@graphql_bp.route('/health', methods=['GET'])
def graphql_health():
    """Health check for GraphQL endpoint"""
    return jsonify({
        'status': 'healthy',
        'endpoint': '/graphql',
        'playground': '/graphql (GET)',
        'schema': '/graphql/schema',
    })
