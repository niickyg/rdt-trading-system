/**
 * RDT Trading System WebSocket Client
 *
 * Provides real-time streaming of:
 * - Trading signals
 * - Position updates
 * - Scanner progress/results
 * - System alerts
 * - Price updates
 */

class RDTWebSocket {
    constructor(options = {}) {
        this.options = {
            autoConnect: true,
            autoReconnect: true,
            reconnectInterval: 3000,
            maxReconnectAttempts: 10,
            pingInterval: 25000,
            debug: false,
            ...options
        };

        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.pingTimer = null;
        this.clientId = null;

        // Room subscriptions
        this.subscribedRooms = new Set();

        // Event handlers
        this.eventHandlers = {
            connect: [],
            disconnect: [],
            error: [],
            signal: [],
            signals_batch: [],
            position: [],
            scanner: [],
            alert: [],
            price: [],
            subscribed: [],
            unsubscribed: [],
            rooms_list: [],
            pong: []
        };

        // Available rooms
        this.availableRooms = ['signals', 'positions', 'scanner', 'alerts', 'prices'];

        if (this.options.autoConnect) {
            this.connect();
        }
    }

    /**
     * Connect to the WebSocket server
     */
    connect() {
        if (this.socket && this.connected) {
            this.log('Already connected');
            return;
        }

        try {
            // Load Socket.IO client library if not already loaded
            if (typeof io === 'undefined') {
                this.loadSocketIO(() => this.initializeSocket());
            } else {
                this.initializeSocket();
            }
        } catch (error) {
            this.log('Connection error:', error);
            this.handleError(error);
        }
    }

    /**
     * Load Socket.IO client library dynamically
     */
    loadSocketIO(callback) {
        const script = document.createElement('script');
        script.src = 'https://cdn.socket.io/4.7.5/socket.io.min.js';
        script.crossOrigin = 'anonymous';
        script.onload = callback;
        script.onerror = () => {
            this.log('Failed to load Socket.IO library');
            this.handleError(new Error('Failed to load Socket.IO library'));
        };
        document.head.appendChild(script);
    }

    /**
     * Initialize the Socket.IO connection
     */
    initializeSocket() {
        const socketUrl = this.options.url || window.location.origin;

        this.socket = io(socketUrl, {
            transports: ['websocket', 'polling'],
            reconnection: false, // We handle reconnection ourselves
            timeout: 20000
        });

        this.setupEventListeners();
    }

    /**
     * Set up Socket.IO event listeners
     */
    setupEventListeners() {
        // Connection events
        this.socket.on('connect', () => {
            this.log('Connected to WebSocket server');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.startPing();
            this.emit('connect', { connected: true });
        });

        this.socket.on('connected', (data) => {
            this.clientId = data.client_id;
            this.log('Received client ID:', this.clientId);

            // Re-subscribe to rooms after reconnection
            if (this.subscribedRooms.size > 0) {
                this.subscribe(Array.from(this.subscribedRooms));
            }
        });

        this.socket.on('disconnect', (reason) => {
            this.log('Disconnected:', reason);
            this.connected = false;
            this.stopPing();
            this.emit('disconnect', { reason });

            if (this.options.autoReconnect) {
                this.scheduleReconnect();
            }
        });

        this.socket.on('connect_error', (error) => {
            this.log('Connection error:', error);
            this.handleError(error);
        });

        // Data events
        this.socket.on('signal', (data) => {
            this.log('Received signal:', data);
            this.emit('signal', data);
        });

        this.socket.on('signals_batch', (data) => {
            this.log('Received signals batch:', data);
            this.emit('signals_batch', data);
        });

        this.socket.on('position', (data) => {
            this.log('Received position update:', data);
            this.emit('position', data);
        });

        this.socket.on('scanner', (data) => {
            this.log('Received scanner update:', data);
            this.emit('scanner', data);
        });

        this.socket.on('alert', (data) => {
            this.log('Received alert:', data);
            this.emit('alert', data);
        });

        this.socket.on('price', (data) => {
            this.log('Received price update:', data);
            this.emit('price', data);
        });

        // Subscription events
        this.socket.on('subscribed', (data) => {
            this.log('Subscribed to rooms:', data.rooms);
            data.rooms.forEach(room => this.subscribedRooms.add(room));
            this.emit('subscribed', data);
        });

        this.socket.on('unsubscribed', (data) => {
            this.log('Unsubscribed from rooms:', data.rooms);
            data.rooms.forEach(room => this.subscribedRooms.delete(room));
            this.emit('unsubscribed', data);
        });

        this.socket.on('rooms_list', (data) => {
            this.log('Rooms list:', data);
            this.emit('rooms_list', data);
        });

        // Ping/pong
        this.socket.on('pong', (data) => {
            this.emit('pong', data);
        });
    }

    /**
     * Disconnect from the WebSocket server
     */
    disconnect() {
        this.options.autoReconnect = false;
        this.stopPing();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }

        this.connected = false;
        this.subscribedRooms.clear();
    }

    /**
     * Subscribe to one or more rooms
     * @param {string|string[]} rooms - Room name(s) to subscribe to
     */
    subscribe(rooms) {
        if (!this.connected) {
            this.log('Not connected, cannot subscribe');
            return;
        }

        const roomList = Array.isArray(rooms) ? rooms : [rooms];
        const validRooms = roomList.filter(r => this.availableRooms.includes(r));

        if (validRooms.length > 0) {
            this.socket.emit('subscribe', { rooms: validRooms });
        }
    }

    /**
     * Unsubscribe from one or more rooms
     * @param {string|string[]} rooms - Room name(s) to unsubscribe from
     */
    unsubscribe(rooms) {
        if (!this.connected) {
            this.log('Not connected, cannot unsubscribe');
            return;
        }

        const roomList = Array.isArray(rooms) ? rooms : [rooms];
        this.socket.emit('unsubscribe', { rooms: roomList });
    }

    /**
     * Subscribe to all available rooms
     */
    subscribeAll() {
        this.subscribe(this.availableRooms);
    }

    /**
     * Unsubscribe from all rooms
     */
    unsubscribeAll() {
        this.unsubscribe(Array.from(this.subscribedRooms));
    }

    /**
     * Get list of currently subscribed rooms
     */
    getRooms() {
        if (this.connected) {
            this.socket.emit('get_rooms');
        }
        return Array.from(this.subscribedRooms);
    }

    /**
     * Add an event handler
     * @param {string} event - Event name
     * @param {function} handler - Event handler function
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
        return this;
    }

    /**
     * Remove an event handler
     * @param {string} event - Event name
     * @param {function} handler - Event handler function to remove
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
        return this;
    }

    /**
     * Emit an event to registered handlers
     */
    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            });
        }
    }

    /**
     * Handle errors
     */
    handleError(error) {
        this.emit('error', { error: error.message || error });

        if (this.options.autoReconnect && !this.connected) {
            this.scheduleReconnect();
        }
    }

    /**
     * Schedule a reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            return;
        }

        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            this.log('Max reconnect attempts reached');
            this.emit('error', { error: 'Max reconnect attempts reached' });
            return;
        }

        this.reconnectAttempts++;
        const delay = this.options.reconnectInterval * this.reconnectAttempts;

        this.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    }

    /**
     * Start ping timer
     */
    startPing() {
        this.stopPing();
        this.pingTimer = setInterval(() => {
            if (this.connected) {
                this.socket.emit('ping');
            }
        }, this.options.pingInterval);
    }

    /**
     * Stop ping timer
     */
    stopPing() {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    /**
     * Log message if debug is enabled
     */
    log(...args) {
        if (this.options.debug) {
            console.log('[RDTWebSocket]', ...args);
        }
    }

    /**
     * Check if connected
     */
    isConnected() {
        return this.connected;
    }

    /**
     * Get connection status
     */
    getStatus() {
        return {
            connected: this.connected,
            clientId: this.clientId,
            subscribedRooms: Array.from(this.subscribedRooms),
            reconnectAttempts: this.reconnectAttempts
        };
    }
}


/**
 * RDT Dashboard WebSocket Manager
 *
 * High-level wrapper for common dashboard operations
 */
class RDTDashboardSocket {
    constructor(options = {}) {
        this.ws = new RDTWebSocket({
            debug: options.debug || false,
            ...options
        });

        // UI elements cache
        this.elements = {};

        // State
        this.signals = [];
        this.positions = [];
        this.alerts = [];
        this.scannerState = null;

        this.setupHandlers();
    }

    /**
     * Set up event handlers
     */
    setupHandlers() {
        this.ws.on('connect', () => {
            this.updateConnectionStatus(true);
        });

        this.ws.on('disconnect', () => {
            this.updateConnectionStatus(false);
        });

        this.ws.on('signal', (data) => {
            this.handleSignal(data);
        });

        this.ws.on('signals_batch', (data) => {
            this.handleSignalsBatch(data);
        });

        this.ws.on('position', (data) => {
            this.handlePosition(data);
        });

        this.ws.on('scanner', (data) => {
            this.handleScanner(data);
        });

        this.ws.on('alert', (data) => {
            this.handleAlert(data);
        });

        this.ws.on('price', (data) => {
            this.handlePrice(data);
        });
    }

    /**
     * Subscribe to all dashboard-relevant rooms
     */
    subscribeAll() {
        this.ws.subscribeAll();
    }

    /**
     * Subscribe to specific rooms
     */
    subscribe(rooms) {
        this.ws.subscribe(rooms);
    }

    /**
     * Update connection status in UI
     */
    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('ws-status-dot');
        const statusText = document.getElementById('ws-status-text');

        if (statusDot) {
            statusDot.className = connected ? 'status-dot online' : 'status-dot offline';
        }

        if (statusText) {
            statusText.textContent = connected ? 'Live' : 'Disconnected';
        }

        // Add a global indicator if it doesn't exist
        this.ensureGlobalIndicator(connected);
    }

    /**
     * Ensure global connection indicator exists
     */
    ensureGlobalIndicator(connected) {
        let indicator = document.getElementById('ws-global-indicator');

        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'ws-global-indicator';
            indicator.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: ${connected ? '#22c55e' : '#ef4444'};
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                z-index: 9999;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                transition: background 0.3s ease;
            `;

            const dot = document.createElement('span');
            dot.id = 'ws-global-dot';
            dot.style.cssText = `
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: white;
                animation: ${connected ? 'pulse 2s infinite' : 'none'};
            `;

            const text = document.createElement('span');
            text.id = 'ws-global-text';
            text.textContent = connected ? 'Live' : 'Disconnected';

            indicator.appendChild(dot);
            indicator.appendChild(text);
            document.body.appendChild(indicator);

            // Add pulse animation
            if (!document.getElementById('ws-pulse-style')) {
                const style = document.createElement('style');
                style.id = 'ws-pulse-style';
                style.textContent = `
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                `;
                document.head.appendChild(style);
            }
        } else {
            indicator.style.background = connected ? '#22c55e' : '#ef4444';
            const dot = document.getElementById('ws-global-dot');
            const text = document.getElementById('ws-global-text');
            if (dot) dot.style.animation = connected ? 'pulse 2s infinite' : 'none';
            if (text) text.textContent = connected ? 'Live' : 'Disconnected';
        }
    }

    /**
     * Handle new signal
     */
    handleSignal(data) {
        const signalData = data.data || data;

        // Add to signals array
        this.signals.unshift(signalData);
        if (this.signals.length > 100) {
            this.signals.pop();
        }

        // Update UI
        this.updateSignalsUI(signalData);

        // Show notification
        this.showNotification('New Signal', `${signalData.symbol} ${signalData.direction.toUpperCase()}`);
    }

    /**
     * Handle signals batch
     */
    handleSignalsBatch(data) {
        const signals = data.data?.signals || [];
        signals.forEach(signal => {
            this.signals.unshift(signal);
        });

        // Trim to 100 signals
        this.signals = this.signals.slice(0, 100);

        // Update UI with all signals
        this.updateSignalsListUI();
    }

    /**
     * Handle position update
     */
    handlePosition(data) {
        const positionData = data.data || data;

        // Update or add position
        const index = this.positions.findIndex(p => p.position_id === positionData.position_id);
        if (index >= 0) {
            this.positions[index] = positionData;
        } else {
            this.positions.push(positionData);
        }

        // Update UI
        this.updatePositionsUI(positionData);
    }

    /**
     * Handle scanner update
     */
    handleScanner(data) {
        this.scannerState = data.data || data;

        // Update scanner UI
        this.updateScannerUI(this.scannerState);
    }

    /**
     * Handle alert
     */
    handleAlert(data) {
        const alertData = data.data || data;

        // Add to alerts
        this.alerts.unshift({
            ...alertData,
            timestamp: data.timestamp || new Date().toISOString()
        });

        // Trim alerts
        if (this.alerts.length > 50) {
            this.alerts.pop();
        }

        // Show alert notification
        this.showAlertNotification(alertData);
    }

    /**
     * Handle price update
     */
    handlePrice(data) {
        const priceData = data.data || data;

        // Update price elements
        const priceEl = document.getElementById(`price-${priceData.symbol}`);
        if (priceEl) {
            priceEl.textContent = `$${priceData.price.toFixed(2)}`;
        }

        const changeEl = document.getElementById(`change-${priceData.symbol}`);
        if (changeEl && priceData.change_pct !== undefined) {
            const sign = priceData.change_pct >= 0 ? '+' : '';
            changeEl.textContent = `${sign}${priceData.change_pct.toFixed(2)}%`;
            changeEl.className = priceData.change_pct >= 0 ? 'text-success' : 'text-danger';
        }
    }

    /**
     * Update signals UI
     */
    updateSignalsUI(signal) {
        const container = document.getElementById('signals-container');
        if (!container) return;

        // Create signal element
        const signalEl = this.createSignalElement(signal);

        // Prepend to container
        const existingEmpty = container.querySelector('.empty-state, .loading-state');
        if (existingEmpty) {
            existingEmpty.remove();
        }

        container.insertBefore(signalEl, container.firstChild);

        // Highlight new signal
        signalEl.style.animation = 'highlightNew 2s ease';

        // Remove old signals if too many
        while (container.children.length > 20) {
            container.lastElementChild.remove();
        }
    }

    /**
     * Update full signals list UI
     */
    updateSignalsListUI() {
        const container = document.getElementById('signals-container');
        if (!container) return;

        if (this.signals.length === 0) {
            container.textContent = '';
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'empty-state';
            const p = document.createElement('p');
            p.textContent = 'No signals yet';
            emptyDiv.appendChild(p);
            container.appendChild(emptyDiv);
            return;
        }

        container.textContent = '';
        this.signals.slice(0, 20).forEach(s => {
            container.appendChild(this.createSignalElement(s));
        });
    }

    /**
     * Create signal HTML element
     */
    _escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    createSignalElement(signal) {
        const dirClass = signal.direction === 'long' ? 'badge-long' : 'badge-short';

        const row = document.createElement('div');
        row.className = 'signal-row';

        const symbolDiv = document.createElement('div');
        symbolDiv.className = 'signal-symbol';
        symbolDiv.textContent = signal.symbol || 'N/A';

        const dirDiv = document.createElement('div');
        dirDiv.className = 'signal-direction';
        const badge = document.createElement('span');
        badge.className = 'badge ' + dirClass;
        badge.textContent = (signal.direction || '').toUpperCase();
        dirDiv.appendChild(badge);

        const pricesDiv = document.createElement('div');
        pricesDiv.className = 'signal-prices';

        const priceItems = [
            { label: 'Entry', value: signal.entry_price, cls: '' },
            { label: 'Stop', value: signal.stop_price, cls: 'text-danger' },
            { label: 'Target', value: signal.target_price, cls: 'text-success' }
        ];

        priceItems.forEach(item => {
            const priceItem = document.createElement('div');
            priceItem.className = 'signal-price-item';
            const labelDiv = document.createElement('div');
            labelDiv.className = 'signal-price-label';
            labelDiv.textContent = item.label;
            const valueDiv = document.createElement('div');
            valueDiv.className = 'signal-price-value' + (item.cls ? ' ' + item.cls : '');
            valueDiv.textContent = '$' + (item.value || 0).toFixed(2);
            priceItem.appendChild(labelDiv);
            priceItem.appendChild(valueDiv);
            pricesDiv.appendChild(priceItem);
        });

        const metaDiv = document.createElement('div');
        metaDiv.className = 'signal-meta';
        const rrsDiv = document.createElement('div');
        rrsDiv.className = 'signal-rrs';
        rrsDiv.textContent = (signal.strategy_name ? signal.strategy_name.replace(/_/g, ' ').toUpperCase() + ' | ' : '') + 'RRS: ' + (signal.rrs || 0).toFixed(2);
        const timeDiv = document.createElement('div');
        timeDiv.className = 'signal-time';
        timeDiv.textContent = this.formatTime(signal.generated_at);
        metaDiv.appendChild(rrsDiv);
        metaDiv.appendChild(timeDiv);

        row.appendChild(symbolDiv);
        row.appendChild(dirDiv);
        row.appendChild(pricesDiv);
        row.appendChild(metaDiv);

        return row;
    }

    /**
     * Update positions UI
     */
    updatePositionsUI(position) {
        const container = document.getElementById('positions-container');
        if (!container) return;

        // Find or create position element
        let positionEl = document.getElementById('position-' + position.position_id);

        if (position.status === 'closed') {
            if (positionEl) {
                positionEl.remove();
            }
            return;
        }

        const newPositionEl = this.createPositionElement(position);

        if (positionEl) {
            positionEl.replaceWith(newPositionEl);
        } else {
            const existingEmpty = container.querySelector('.empty-state');
            if (existingEmpty) {
                existingEmpty.remove();
            }
            container.appendChild(newPositionEl);
        }
    }

    /**
     * Create position DOM element
     */
    createPositionElement(position) {
        const pnlClass = position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
        const pnlSign = position.unrealized_pnl >= 0 ? '+' : '';

        const card = document.createElement('div');
        card.className = 'position-card';
        card.id = 'position-' + (position.position_id || '');

        const infoDiv = document.createElement('div');
        infoDiv.className = 'position-info';

        const dirBadge = document.createElement('span');
        dirBadge.className = 'badge ' + (position.direction === 'long' ? 'badge-long' : 'badge-short');
        dirBadge.textContent = (position.direction || '').toUpperCase();

        const symbolStrong = document.createElement('strong');
        symbolStrong.textContent = position.symbol || '';

        const detailSpan = document.createElement('span');
        detailSpan.textContent = (position.quantity || '') + ' shares @ $' + (position.entry_price || 0).toFixed(2);

        infoDiv.appendChild(dirBadge);
        infoDiv.appendChild(symbolStrong);
        infoDiv.appendChild(detailSpan);

        const pnlDiv = document.createElement('div');
        pnlDiv.className = 'position-pnl ' + pnlClass;
        pnlDiv.textContent = pnlSign + '$' + (position.unrealized_pnl || 0).toFixed(2) +
            ' (' + pnlSign + (position.unrealized_pnl_pct || 0).toFixed(2) + '%)';

        card.appendChild(infoDiv);
        card.appendChild(pnlDiv);

        return card;
    }

    /**
     * Update scanner UI
     */
    updateScannerUI(scanner) {
        // Update status
        const statusDot = document.getElementById('scanner-status-dot');
        const statusText = document.getElementById('scanner-status');

        if (statusDot && statusText) {
            switch (scanner.status) {
                case 'started':
                case 'progress':
                    statusDot.className = 'status-dot warning';
                    statusText.textContent = 'Scanning...';
                    break;
                case 'completed':
                    statusDot.className = 'status-dot online';
                    statusText.textContent = 'Ready';
                    break;
                case 'error':
                    statusDot.className = 'status-dot offline';
                    statusText.textContent = 'Error';
                    break;
            }
        }

        // Update progress
        if (scanner.progress !== undefined) {
            const progressBar = document.getElementById('scanner-progress');
            if (progressBar) {
                progressBar.style.width = `${scanner.progress}%`;
            }
        }

        // Update counts
        if (scanner.strong_rs_count !== undefined) {
            const rsCount = document.getElementById('strong-rs-count');
            if (rsCount) rsCount.textContent = scanner.strong_rs_count;
        }

        if (scanner.strong_rw_count !== undefined) {
            const rwCount = document.getElementById('strong-rw-count');
            if (rwCount) rwCount.textContent = scanner.strong_rw_count;
        }

        // Update results if completed
        if (scanner.status === 'completed') {
            this.updateScanResults(scanner);
        }
    }

    /**
     * Update scan results
     */
    updateScanResults(scanner) {
        // Update RS results
        const rsContainer = document.getElementById('rs-results');
        if (rsContainer && scanner.strong_rs) {
            rsContainer.textContent = '';
            if (scanner.strong_rs.length === 0) {
                const emptyDiv = document.createElement('div');
                emptyDiv.className = 'empty-state';
                const p = document.createElement('p');
                p.textContent = 'No strong RS stocks found';
                emptyDiv.appendChild(p);
                rsContainer.appendChild(emptyDiv);
            } else {
                scanner.strong_rs.forEach(stock => {
                    const row = document.createElement('div');
                    row.className = 'stock-row';
                    const symbolDiv = document.createElement('div');
                    symbolDiv.className = 'stock-symbol';
                    symbolDiv.textContent = stock.symbol;
                    const rrsDiv = document.createElement('div');
                    rrsDiv.className = 'stock-rrs';
                    const rrsValue = document.createElement('div');
                    rrsValue.className = 'stock-rrs-value strong';
                    rrsValue.textContent = (stock.rrs || 0).toFixed(2);
                    rrsDiv.appendChild(rrsValue);
                    row.appendChild(symbolDiv);
                    row.appendChild(rrsDiv);
                    rsContainer.appendChild(row);
                });
            }
        }

        // Update RW results
        const rwContainer = document.getElementById('rw-results');
        if (rwContainer && scanner.strong_rw) {
            rwContainer.textContent = '';
            if (scanner.strong_rw.length === 0) {
                const emptyDiv = document.createElement('div');
                emptyDiv.className = 'empty-state';
                const p = document.createElement('p');
                p.textContent = 'No strong RW stocks found';
                emptyDiv.appendChild(p);
                rwContainer.appendChild(emptyDiv);
            } else {
                scanner.strong_rw.forEach(stock => {
                    const row = document.createElement('div');
                    row.className = 'stock-row';
                    const symbolDiv = document.createElement('div');
                    symbolDiv.className = 'stock-symbol';
                    symbolDiv.textContent = stock.symbol;
                    const rrsDiv = document.createElement('div');
                    rrsDiv.className = 'stock-rrs';
                    const rrsValue = document.createElement('div');
                    rrsValue.className = 'stock-rrs-value weak';
                    rrsValue.textContent = (stock.rrs || 0).toFixed(2);
                    rrsDiv.appendChild(rrsValue);
                    row.appendChild(symbolDiv);
                    row.appendChild(rrsDiv);
                    rwContainer.appendChild(row);
                });
            }
        }
    }

    /**
     * Show alert notification
     */
    showAlertNotification(alert) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `ws-notification ws-notification-${alert.severity || 'info'}`;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: #1f2937;
            border-left: 4px solid ${this.getSeverityColor(alert.severity)};
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            z-index: 10000;
            max-width: 350px;
            animation: slideIn 0.3s ease;
        `;

        const titleEl = document.createElement('strong');
        titleEl.style.cssText = 'display: block; margin-bottom: 5px;';
        titleEl.textContent = alert.title || 'Alert';
        const messageEl = document.createElement('span');
        messageEl.style.color = '#9ca3af';
        messageEl.textContent = alert.message || '';
        notification.appendChild(titleEl);
        notification.appendChild(messageEl);

        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    /**
     * Show simple notification
     */
    showNotification(title, message) {
        this.showAlertNotification({
            title,
            message,
            severity: 'info'
        });
    }

    /**
     * Get color for severity level
     */
    getSeverityColor(severity) {
        const colors = {
            low: '#22c55e',
            medium: '#f59e0b',
            high: '#ef4444',
            critical: '#dc2626',
            info: '#3b82f6'
        };
        return colors[severity] || colors.info;
    }

    /**
     * Format timestamp
     */
    formatTime(timestamp) {
        if (!timestamp) return 'N/A';
        try {
            return new Date(timestamp).toLocaleTimeString();
        } catch (e) {
            return timestamp;
        }
    }

    /**
     * Get WebSocket status
     */
    getStatus() {
        return this.ws.getStatus();
    }

    /**
     * Disconnect
     */
    disconnect() {
        this.ws.disconnect();
    }
}


// Add CSS for animations
(function() {
    if (!document.getElementById('ws-animations')) {
        const style = document.createElement('style');
        style.id = 'ws-animations';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            @keyframes highlightNew {
                0% { background: rgba(34, 197, 94, 0.2); }
                100% { background: transparent; }
            }
        `;
        document.head.appendChild(style);
    }
})();


// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RDTWebSocket, RDTDashboardSocket };
}
