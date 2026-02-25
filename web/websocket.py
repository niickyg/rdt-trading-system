"""
WebSocket Support for RDT Trading System

Provides real-time streaming of:
- Trading signals
- Position updates
- Scanner progress/results
- System alerts
- Price updates
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import wraps
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from flask_login import current_user
from loguru import logger

from utils.timezone import format_timestamp

# SocketIO instance - will be initialized by init_websocket()
socketio: Optional[SocketIO] = None

# Tier hierarchy for access control (higher index = more access)
TIER_LEVELS = {
    'free': 0,
    'basic': 1,
    'pro': 2,
    'elite': 3,
}

# Default tier for clients without tier info (backward compatibility)
DEFAULT_TIER = 'pro'

# Room names for different data streams
ROOM_SIGNALS = 'signals'
ROOM_POSITIONS = 'positions'
ROOM_SCANNER = 'scanner'
ROOM_ALERTS = 'alerts'
ROOM_PRICES = 'prices'

# All available rooms
AVAILABLE_ROOMS = [ROOM_SIGNALS, ROOM_POSITIONS, ROOM_SCANNER, ROOM_ALERTS, ROOM_PRICES]

# Track connected clients
connected_clients: Dict[str, Dict] = {}


def init_websocket(app, async_mode='eventlet'):
    """
    Initialize Flask-SocketIO with the Flask application.

    Args:
        app: Flask application instance
        async_mode: Async mode to use ('eventlet', 'gevent', or 'threading')

    Returns:
        SocketIO instance
    """
    global socketio

    # SECURITY: Configure CORS with restricted origins
    # Do NOT use "*" in production
    websocket_cors_origins = os.environ.get(
        'WEBSOCKET_CORS_ORIGINS',
        'http://localhost:5000,http://127.0.0.1:5000'
    ).split(',')
    websocket_cors_origins = [origin.strip() for origin in websocket_cors_origins if origin.strip()]

    socketio = SocketIO(
        app,
        async_mode=async_mode,
        cors_allowed_origins=websocket_cors_origins,
        ping_timeout=60,
        ping_interval=25,
        logger=False,
        engineio_logger=False
    )

    # Register event handlers
    register_event_handlers(socketio)

    logger.info(f"WebSocket initialized with async_mode={async_mode}")

    return socketio


def register_event_handlers(sio: SocketIO):
    """Register all WebSocket event handlers."""

    @sio.on('connect')
    def handle_connect():
        """Handle client connection with authentication."""
        client_id = request.sid

        # Check authentication - API key from headers only
        api_key = request.headers.get('X-API-Key')

        user_info = None
        subscription_tier = None
        if api_key:
            try:
                from api.v1.auth import api_key_manager
                user = api_key_manager.get_user_by_api_key(api_key)
                if user and user.is_active and not user.is_expired:
                    user_info = user
                    subscription_tier = user.subscription_tier.value
            except Exception as e:
                logger.warning(f"WebSocket auth error: {e}")

        # Also allow Flask-Login authenticated users (dashboard)
        if not user_info:
            if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
                user_info = current_user
            else:
                client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                if client_ip and ',' in client_ip:
                    client_ip = client_ip.split(',')[0].strip()
                logger.warning(f"WebSocket connection rejected: no valid authentication from {client_ip}")
                return False

        # Store client info
        connected_clients[client_id] = {
            'connected_at': datetime.now().isoformat(),
            'rooms': [],
            'user_id': getattr(user_info, 'id', getattr(user_info, 'user_id', None)),
            'ip_address': request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip() if request.headers.get('X-Forwarded-For') else request.remote_addr,
            'subscription_tier': subscription_tier
        }

        logger.info(f"WebSocket client connected: {client_id}")

        emit('connected', {
            'client_id': client_id,
            'timestamp': format_timestamp(),
            'available_rooms': AVAILABLE_ROOMS,
            'message': 'Connected to RDT Trading WebSocket'
        })

    @sio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        client_id = request.sid

        if client_id in connected_clients:
            client_info = connected_clients.pop(client_id)
            logger.info(f"WebSocket client disconnected: {client_id}, rooms: {client_info.get('rooms', [])}")
        else:
            logger.info(f"WebSocket client disconnected: {client_id}")

    @sio.on('subscribe')
    def handle_subscribe(data):
        """
        Handle client subscription to rooms.

        Expected data format:
        {
            "rooms": ["signals", "positions", "scanner", "alerts"]
        }
        or
        {
            "room": "signals"
        }
        """
        client_id = request.sid

        # Handle both single room and multiple rooms
        if isinstance(data, dict):
            room_list = data.get('rooms', [])
            if not room_list and 'room' in data:
                room_list = [data['room']]
        elif isinstance(data, str):
            room_list = [data]
        else:
            room_list = []

        subscribed = []
        invalid = []

        for room in room_list:
            if room in AVAILABLE_ROOMS:
                join_room(room)
                subscribed.append(room)

                # Update client tracking
                if client_id in connected_clients:
                    if room not in connected_clients[client_id]['rooms']:
                        connected_clients[client_id]['rooms'].append(room)

                logger.debug(f"Client {client_id} subscribed to room: {room}")
            else:
                invalid.append(room)

        emit('subscribed', {
            'rooms': subscribed,
            'invalid_rooms': invalid,
            'timestamp': format_timestamp(),
            'message': f'Subscribed to {len(subscribed)} room(s)'
        })

    @sio.on('unsubscribe')
    def handle_unsubscribe(data):
        """
        Handle client unsubscription from rooms.

        Expected data format:
        {
            "rooms": ["signals", "positions"]
        }
        or
        {
            "room": "signals"
        }
        """
        client_id = request.sid

        # Handle both single room and multiple rooms
        if isinstance(data, dict):
            room_list = data.get('rooms', [])
            if not room_list and 'room' in data:
                room_list = [data['room']]
        elif isinstance(data, str):
            room_list = [data]
        else:
            room_list = []

        unsubscribed = []

        for room in room_list:
            if room in AVAILABLE_ROOMS:
                leave_room(room)
                unsubscribed.append(room)

                # Update client tracking
                if client_id in connected_clients:
                    if room in connected_clients[client_id]['rooms']:
                        connected_clients[client_id]['rooms'].remove(room)

                logger.debug(f"Client {client_id} unsubscribed from room: {room}")

        emit('unsubscribed', {
            'rooms': unsubscribed,
            'timestamp': format_timestamp(),
            'message': f'Unsubscribed from {len(unsubscribed)} room(s)'
        })

    @sio.on('ping')
    def handle_ping():
        """Handle client ping for connection keepalive."""
        emit('pong', {
            'timestamp': format_timestamp()
        })

    @sio.on('get_rooms')
    def handle_get_rooms():
        """Return list of rooms client is subscribed to."""
        client_id = request.sid
        current_rooms = []

        if client_id in connected_clients:
            current_rooms = connected_clients[client_id].get('rooms', [])

        emit('rooms_list', {
            'subscribed_rooms': current_rooms,
            'available_rooms': AVAILABLE_ROOMS,
            'timestamp': format_timestamp()
        })

    @sio.on('error')
    def handle_error(error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")


# =============================================================================
# TIER FILTERING HELPERS
# =============================================================================

def _get_client_tier(client_info: Dict) -> str:
    """
    Get the effective tier for a client.

    Returns the client's subscription tier, defaulting to DEFAULT_TIER
    for backward compatibility when tier info is not available.
    """
    tier = client_info.get('subscription_tier')
    if tier is None:
        return DEFAULT_TIER
    return tier


def _get_clients_in_room_by_tier(room: str, min_tier: Optional[str] = None) -> List[str]:
    """
    Get session IDs of clients subscribed to a room, optionally filtered by minimum tier.

    Args:
        room: The room name to check
        min_tier: Minimum subscription tier required (None means all tiers)

    Returns:
        List of session IDs that match the criteria
    """
    min_level = TIER_LEVELS.get(min_tier, 0) if min_tier else 0
    sids = []
    for sid, client_info in connected_clients.items():
        if room in client_info.get('rooms', []):
            client_tier = _get_client_tier(client_info)
            client_level = TIER_LEVELS.get(client_tier, TIER_LEVELS[DEFAULT_TIER])
            if client_level >= min_level:
                sids.append(sid)
    return sids


# =============================================================================
# BROADCASTING FUNCTIONS
# =============================================================================

def broadcast_signal(signal_data: Dict):
    """
    Broadcast a new trading signal to the 'signals' room.

    Args:
        signal_data: Signal data dictionary containing:
            - symbol: Stock ticker
            - direction: 'long' or 'short'
            - strength: 'strong', 'moderate', 'weak'
            - rrs: RRS score
            - entry_price: Entry price
            - stop_price: Stop loss price
            - target_price: Target price
            - atr: ATR value
            - generated_at: Timestamp
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast signal")
        return

    timestamp = format_timestamp()
    confidence = signal_data.get('confidence', 0.0)
    sent_count = 0

    for sid, client_info in connected_clients.items():
        if ROOM_SIGNALS not in client_info.get('rooms', []):
            continue

        tier = _get_client_tier(client_info)

        # FREE tier: only signals with confidence >= 0.8
        if tier == 'free' and confidence < 0.8:
            continue

        payload = {
            'event': 'new_signal',
            'data': signal_data.copy(),
            'timestamp': timestamp
        }

        # ELITE tier: include extra metadata
        if tier == 'elite':
            payload['data']['_elite_metadata'] = {
                'delivery_tier': 'elite',
                'full_analytics': True,
            }

        socketio.emit('signal', payload, to=sid)
        sent_count += 1

    logger.debug(f"Broadcasted signal: {signal_data.get('symbol')} {signal_data.get('direction')} to {sent_count} client(s)")


def broadcast_position_update(position_data: Dict):
    """
    Broadcast a position update to the 'positions' room.

    Args:
        position_data: Position data dictionary containing:
            - position_id: Unique position identifier
            - symbol: Stock ticker
            - direction: 'long' or 'short'
            - quantity: Number of shares
            - entry_price: Entry price
            - current_price: Current price
            - unrealized_pnl: Unrealized P&L
            - unrealized_pnl_pct: Unrealized P&L percentage
            - status: 'open', 'closed', 'partial'
            - updated_at: Timestamp
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast position update")
        return

    payload = {
        'event': 'position_update',
        'data': position_data,
        'timestamp': format_timestamp()
    }

    # Position data is premium — only PRO and ELITE tiers
    eligible_sids = _get_clients_in_room_by_tier(ROOM_POSITIONS, min_tier='pro')
    for sid in eligible_sids:
        socketio.emit('position', payload, to=sid)

    logger.debug(f"Broadcasted position update: {position_data.get('symbol')} to {len(eligible_sids)} client(s)")


def broadcast_scanner_update(scanner_data: Dict):
    """
    Broadcast scanner progress/results to the 'scanner' room.

    Args:
        scanner_data: Scanner data dictionary containing:
            - status: 'started', 'progress', 'completed', 'error'
            - progress: Progress percentage (0-100) if status is 'progress'
            - symbols_scanned: Number of symbols scanned
            - total_symbols: Total symbols to scan
            - strong_rs: List of strong RS stocks (if completed)
            - strong_rw: List of strong RW stocks (if completed)
            - message: Optional status message
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast scanner update")
        return

    payload = {
        'event': 'scanner_update',
        'data': scanner_data,
        'timestamp': format_timestamp()
    }

    # Scanner status updates are available to all tiers
    socketio.emit('scanner', payload, room=ROOM_SCANNER)
    logger.debug(f"Broadcasted scanner update: {scanner_data.get('status')}")


def broadcast_alert(alert_data: Dict):
    """
    Broadcast an alert to the 'alerts' room.

    Args:
        alert_data: Alert data dictionary containing:
            - alert_type: 'signal', 'price', 'system', 'error', 'info', 'warning'
            - severity: 'low', 'medium', 'high', 'critical'
            - title: Alert title
            - message: Alert message
            - symbol: Optional stock ticker
            - data: Optional additional data
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast alert")
        return

    payload = {
        'event': 'alert',
        'data': alert_data,
        'timestamp': format_timestamp()
    }

    # Alerts are available to all tiers
    socketio.emit('alert', payload, room=ROOM_ALERTS)
    logger.debug(f"Broadcasted alert: {alert_data.get('alert_type')} - {alert_data.get('title')}")


def broadcast_price_update(symbol: str, price: float, change: Optional[float] = None,
                           change_pct: Optional[float] = None, volume: Optional[int] = None):
    """
    Broadcast a price update to the 'prices' room.

    Args:
        symbol: Stock ticker symbol
        price: Current price
        change: Price change (optional)
        change_pct: Price change percentage (optional)
        volume: Trading volume (optional)
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast price update")
        return

    price_data = {
        'symbol': symbol,
        'price': price,
        'change': change,
        'change_pct': change_pct,
        'volume': volume
    }

    payload = {
        'event': 'price_update',
        'data': price_data,
        'timestamp': format_timestamp()
    }

    # Price updates are available to all tiers
    socketio.emit('price', payload, room=ROOM_PRICES)


def broadcast_multiple_signals(signals: List[Dict]):
    """
    Broadcast multiple signals at once.

    Args:
        signals: List of signal data dictionaries
    """
    if socketio is None:
        logger.warning("WebSocket not initialized, cannot broadcast signals")
        return

    timestamp = format_timestamp()
    sent_count = 0

    for sid, client_info in connected_clients.items():
        if ROOM_SIGNALS not in client_info.get('rooms', []):
            continue

        tier = _get_client_tier(client_info)

        # FREE tier: only include signals with confidence >= 0.8
        if tier == 'free':
            filtered = [s for s in signals if s.get('confidence', 0.0) >= 0.8]
        else:
            filtered = list(signals)

        if not filtered:
            continue

        payload = {
            'event': 'signals_batch',
            'data': {
                'signals': filtered,
                'count': len(filtered)
            },
            'timestamp': timestamp
        }

        # ELITE tier: include extra metadata
        if tier == 'elite':
            payload['data']['_elite_metadata'] = {
                'delivery_tier': 'elite',
                'full_analytics': True,
            }

        socketio.emit('signals_batch', payload, to=sid)
        sent_count += 1

    logger.debug(f"Broadcasted {len(signals)} signals in batch to {sent_count} client(s)")


def broadcast_scan_started(watchlist_size: int):
    """Broadcast that a scan has started."""
    broadcast_scanner_update({
        'status': 'started',
        'total_symbols': watchlist_size,
        'symbols_scanned': 0,
        'progress': 0,
        'message': f'Starting scan of {watchlist_size} symbols'
    })


def broadcast_scan_progress(symbols_scanned: int, total_symbols: int, current_symbol: Optional[str] = None):
    """Broadcast scan progress."""
    progress = int((symbols_scanned / total_symbols) * 100) if total_symbols > 0 else 0

    broadcast_scanner_update({
        'status': 'progress',
        'symbols_scanned': symbols_scanned,
        'total_symbols': total_symbols,
        'progress': progress,
        'current_symbol': current_symbol,
        'message': f'Scanned {symbols_scanned}/{total_symbols} symbols ({progress}%)'
    })


def broadcast_scan_completed(strong_rs: List[Dict], strong_rw: List[Dict],
                             total_scanned: int, duration_seconds: float):
    """Broadcast that a scan has completed."""
    broadcast_scanner_update({
        'status': 'completed',
        'symbols_scanned': total_scanned,
        'total_symbols': total_scanned,
        'progress': 100,
        'strong_rs_count': len(strong_rs),
        'strong_rw_count': len(strong_rw),
        'strong_rs': strong_rs[:10],  # Top 10 RS
        'strong_rw': strong_rw[:10],  # Top 10 RW
        'duration_seconds': round(duration_seconds, 1),
        'message': f'Scan completed: {len(strong_rs)} RS, {len(strong_rw)} RW stocks found'
    })

    # Also broadcast individual signals for the strongest finds
    for signal in strong_rs[:5]:
        signal_data = {
            'symbol': signal.get('symbol'),
            'direction': 'long',
            'strength': 'strong' if signal.get('rrs', 0) > 2.5 else 'moderate',
            'rrs': signal.get('rrs'),
            'entry_price': signal.get('price'),
            'stop_price': signal.get('price', 0) - signal.get('atr', 0),
            'target_price': signal.get('price', 0) + (signal.get('atr', 0) * 2),
            'atr': signal.get('atr'),
            'stock_change_pct': signal.get('stock_pc'),
            'spy_change_pct': signal.get('spy_pc'),
            'generated_at': format_timestamp()
        }
        broadcast_signal(signal_data)

    for signal in strong_rw[:5]:
        signal_data = {
            'symbol': signal.get('symbol'),
            'direction': 'short',
            'strength': 'strong' if signal.get('rrs', 0) < -2.5 else 'moderate',
            'rrs': signal.get('rrs'),
            'entry_price': signal.get('price'),
            'stop_price': signal.get('price', 0) + signal.get('atr', 0),
            'target_price': signal.get('price', 0) - (signal.get('atr', 0) * 2),
            'atr': signal.get('atr'),
            'stock_change_pct': signal.get('stock_pc'),
            'spy_change_pct': signal.get('spy_pc'),
            'generated_at': format_timestamp()
        }
        broadcast_signal(signal_data)


def broadcast_scan_error(error_message: str):
    """Broadcast a scan error."""
    broadcast_scanner_update({
        'status': 'error',
        'message': error_message
    })

    broadcast_alert({
        'alert_type': 'error',
        'severity': 'high',
        'title': 'Scanner Error',
        'message': error_message
    })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_connected_clients() -> Dict[str, Dict]:
    """Get information about connected clients."""
    return connected_clients.copy()


def get_clients_in_room(room: str) -> int:
    """Get the number of clients in a specific room."""
    count = 0
    for client_id, client_info in connected_clients.items():
        if room in client_info.get('rooms', []):
            count += 1
    return count


def get_room_stats() -> Dict[str, int]:
    """Get statistics about room subscriptions."""
    stats = {room: 0 for room in AVAILABLE_ROOMS}
    stats['total_connected'] = len(connected_clients)

    for client_id, client_info in connected_clients.items():
        for room in client_info.get('rooms', []):
            if room in stats:
                stats[room] += 1

    return stats


def broadcast_to_all(event: str, data: Any):
    """Broadcast a message to all connected clients."""
    if socketio is None:
        return

    payload = {
        'event': event,
        'data': data,
        'timestamp': format_timestamp()
    }

    socketio.emit(event, payload)


def broadcast_system_message(message: str, severity: str = 'info'):
    """Broadcast a system message to all clients."""
    broadcast_alert({
        'alert_type': 'system',
        'severity': severity,
        'title': 'System Message',
        'message': message
    })
