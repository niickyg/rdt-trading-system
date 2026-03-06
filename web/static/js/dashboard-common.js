/**
 * Dashboard Common Utilities
 * Single source of truth for strategy constants, formatters, and Chart.js helpers.
 */

// --- Toast Notification System ---
var TOAST_ICONS = TOAST_ICONS || {
    success: '&#10003;',
    error: '&#10007;',
    warning: '&#9888;',
    info: '&#8505;'
};

function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;
    var container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    // Cap at 5 visible toasts
    while (container.children.length >= 5) {
        container.removeChild(container.firstChild);
    }
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML =
        '<span class="toast-icon">' + (TOAST_ICONS[type] || TOAST_ICONS.info) + '</span>' +
        '<span class="toast-body">' + escapeHtml(message) + '</span>' +
        '<button class="toast-close" aria-label="Close">&times;</button>';
    toast.querySelector('.toast-close').onclick = function() { dismissToast(toast); };
    container.appendChild(toast);
    var timer = setTimeout(function() { dismissToast(toast); }, duration);
    toast._timer = timer;
    return toast;
}

function dismissToast(toast) {
    if (toast._dismissed) return;
    toast._dismissed = true;
    clearTimeout(toast._timer);
    toast.classList.add('removing');
    setTimeout(function() { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 260);
}

// --- Skeleton Loading Generators ---
function skeletonStats(n) {
    var html = '';
    for (var i = 0; i < n; i++) {
        html += '<div class="skeleton-stat"></div>';
    }
    return html;
}

function skeletonRows(n) {
    var html = '';
    for (var i = 0; i < n; i++) {
        html += '<tr><td colspan="13"><div class="skeleton-row">' +
            '<span class="skeleton sk-symbol"></span>' +
            '<span class="skeleton sk-badge"></span>' +
            '<span class="skeleton sk-price"></span>' +
            '<span class="skeleton sk-price"></span>' +
            '<span class="skeleton sk-price"></span>' +
            '<span class="skeleton sk-metric"></span>' +
            '<span class="skeleton sk-metric"></span>' +
            '<span class="skeleton sk-time"></span>' +
            '</div></td></tr>';
    }
    return html;
}

function skeletonCards(n) {
    var html = '';
    for (var i = 0; i < n; i++) {
        html += '<div class="skeleton-card">' +
            '<div class="skeleton-card-header"><span class="skeleton" style="width:80px;height:20px"></span><span class="skeleton" style="width:60px;height:20px"></span></div>' +
            '<div class="skeleton-card-details">' +
            '<span class="skeleton" style="width:100%;height:14px"></span>'.repeat(5) +
            '</div></div>';
    }
    return html;
}

function skeletonPills(n) {
    var html = '';
    for (var i = 0; i < n; i++) {
        html += '<span class="skeleton skeleton-pill" style="width:' + (80 + Math.random() * 40) + 'px"></span> ';
    }
    return html;
}

function skeletonChart() {
    return '<div class="skeleton-chart"></div>';
}

// --- Strategy Constants ---
const STRATEGY_COLORS = {
    'rrs_momentum': '#3b82f6',
    'rsi2_mean_reversion': '#22c55e',
    'trend_breakout': '#f59e0b',
    'pead': '#a855f7',
    'gap_fill': '#ef4444',
};

const STRATEGY_LABELS = {
    'rrs_momentum': 'RRS Momentum',
    'rsi2_mean_reversion': 'RSI(2) Mean Rev',
    'trend_breakout': 'Trend Breakout',
    'pead': 'Post-Earnings',
    'gap_fill': 'Gap Fill',
};

const STRATEGY_SHORT = {
    'rrs_momentum': 'RRS',
    'rsi2_mean_reversion': 'RSI2',
    'trend_breakout': 'Trend',
    'pead': 'PEAD',
    'gap_fill': 'Gap Fill',
};

const STATE_COLORS = {
    'running': '#22c55e',
    'paused': '#f59e0b',
    'error': '#ef4444',
    'stopped': '#6b7280',
    'created': '#6b7280',
    'not_started': '#6b7280',
    'initializing': '#3b82f6',
    'stopping': '#f59e0b',
};

// --- Number Count-Up Animation ---
function animateValue(element, start, end, duration, formatter) {
    if (!element || start === end) return;
    duration = duration || 600;
    formatter = formatter || function(v) { return v.toFixed(0); };
    var startTime = null;
    function easeOutExpo(t) {
        return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }
    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        var progress = Math.min((timestamp - startTime) / duration, 1);
        var easedProgress = easeOutExpo(progress);
        var current = start + (end - start) * easedProgress;
        element.textContent = formatter(current);
        if (progress < 1) {
            requestAnimationFrame(step);
        } else {
            element.textContent = formatter(end);
            element.classList.add('metric-updated');
            setTimeout(function() { element.classList.remove('metric-updated'); }, 350);
        }
    }
    requestAnimationFrame(step);
}

function parseStatValue(text) {
    if (!text) return NaN;
    text = text.replace(/[$,%+]/g, '').replace(/,/g, '').trim();
    return parseFloat(text);
}

function animateStatElement(el, newValue, formatter) {
    var oldValue = parseStatValue(el.textContent);
    if (isNaN(oldValue) || isNaN(newValue) || oldValue === newValue) {
        el.textContent = formatter(newValue);
        return;
    }
    animateValue(el, oldValue, newValue, 600, formatter);
}

// --- Session-aware fetch wrapper ---
/**
 * Fetch wrapper that handles session expiry (302 redirect to login).
 * Shows a banner and redirects to /login when session is lost.
 * Use this instead of raw fetch() for all dashboard API calls.
 */
async function dashboardFetch(url, options) {
    const resp = await fetch(url, options || {});
    // Detect login redirect (session expired)
    if (resp.redirected && resp.url.includes('/login')) {
        _showSessionExpiredBanner();
        throw new Error('Session expired');
    }
    // Also catch explicit 401/403
    if (resp.status === 401 || resp.status === 403) {
        _showSessionExpiredBanner();
        throw new Error('Session expired');
    }
    return resp;
}

let _sessionBannerShown = false;
function _showSessionExpiredBanner() {
    if (_sessionBannerShown) return;
    _sessionBannerShown = true;
    const banner = document.createElement('div');
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#ef4444;color:white;padding:12px 20px;text-align:center;font-weight:500;font-size:0.95rem;';
    banner.innerHTML = 'Session expired. <a href="/login" style="color:white;text-decoration:underline;font-weight:700;">Log in again</a>';
    document.body.prepend(banner);
}

// --- Formatters ---
function formatCurrency(value) {
    const sign = value >= 0 ? '' : '-';
    return sign + '$' + Math.abs(value).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatPct(value) {
    const sign = value >= 0 ? '+' : '';
    return sign + value.toFixed(2) + '%';
}

function formatStrategyBadge(name) {
    const color = STRATEGY_COLORS[name] || '#6b7280';
    const label = STRATEGY_SHORT[name] || name;
    return '<span class="badge-strategy" style="background:' + color + '">' + escapeHtml(label) + '</span>';
}

function timeAgo(date) {
    if (typeof date === 'string') date = new Date(date);
    const seconds = Math.floor((new Date() - date) / 1000);
    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + 'm ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + 'h ago';
    const days = Math.floor(hours / 24);
    return days + 'd ago';
}

function escapeHtml(str) {
    if (str == null || typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// --- Chart.js Helpers ---
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: '#9ca3af',
                font: { size: 12 },
                padding: 15,
                usePointStyle: true,
            }
        },
        tooltip: {
            backgroundColor: '#1f2937',
            titleColor: '#e5e7eb',
            bodyColor: '#d1d5db',
            borderColor: '#334155',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 6,
            titleFont: { weight: '600' },
        }
    },
    scales: {
        x: {
            grid: { color: 'rgba(255,255,255,0.07)' },
            ticks: { color: '#9ca3af' },
        },
        y: {
            grid: { color: 'rgba(255,255,255,0.07)' },
            ticks: { color: '#9ca3af' },
        }
    }
};

/**
 * Create a Chart.js chart with consistent dashboard styling.
 * Returns null gracefully if Chart.js failed to load from CDN.
 */
function createChart(canvasId, type, data, extraOptions) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Guard against Chart.js CDN failure
    if (typeof Chart === 'undefined') {
        ctx.parentElement.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-light);font-size:0.85rem;">Chart library unavailable</div>';
        return null;
    }

    const options = Object.assign({}, CHART_DEFAULTS, extraOptions || {});
    // Deep merge plugins
    if (extraOptions && extraOptions.plugins) {
        options.plugins = Object.assign({}, CHART_DEFAULTS.plugins, extraOptions.plugins);
        if (extraOptions.plugins.legend) {
            options.plugins.legend = Object.assign({}, CHART_DEFAULTS.plugins.legend, extraOptions.plugins.legend);
        }
        if (extraOptions.plugins.tooltip) {
            options.plugins.tooltip = Object.assign({}, CHART_DEFAULTS.plugins.tooltip, extraOptions.plugins.tooltip);
        }
    }
    // Deep merge scales
    if (extraOptions && extraOptions.scales) {
        options.scales = options.scales || {};
        for (const axis of Object.keys(extraOptions.scales)) {
            options.scales[axis] = Object.assign({}, (CHART_DEFAULTS.scales || {})[axis] || {}, extraOptions.scales[axis]);
            if (extraOptions.scales[axis].grid) {
                options.scales[axis].grid = Object.assign({}, ((CHART_DEFAULTS.scales || {})[axis] || {}).grid || {}, extraOptions.scales[axis].grid);
            }
            if (extraOptions.scales[axis].ticks) {
                options.scales[axis].ticks = Object.assign({}, ((CHART_DEFAULTS.scales || {})[axis] || {}).ticks || {}, extraOptions.scales[axis].ticks);
            }
        }
    }

    return new Chart(ctx, {
        type: type,
        data: data,
        options: options,
    });
}
