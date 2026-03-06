/**
 * Dashboard Common Utilities
 * Single source of truth for strategy constants, formatters, and Chart.js helpers.
 */

// --- Strategy Constants ---
const STRATEGY_COLORS = {
    'rrs_momentum': '#3498db',
    'rsi2_mean_reversion': '#2ecc71',
    'trend_breakout': '#e67e22',
    'pead': '#9b59b6',
    'gap_fill': '#e74c3c',
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
    'running': '#2ecc71',
    'paused': '#f39c12',
    'error': '#e74c3c',
    'stopped': '#95a5a6',
    'created': '#95a5a6',
    'not_started': '#95a5a6',
    'initializing': '#3498db',
    'stopping': '#f39c12',
};

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
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#e74c3c;color:white;padding:12px 20px;text-align:center;font-weight:500;font-size:0.95rem;';
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
    const color = STRATEGY_COLORS[name] || '#95a5a6';
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
                color: '#666',
                font: { size: 12 },
                padding: 15,
                usePointStyle: true,
            }
        },
        tooltip: {
            backgroundColor: 'rgba(26, 26, 46, 0.95)',
            titleColor: '#fff',
            bodyColor: '#ddd',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { weight: '600' },
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

    return new Chart(ctx, {
        type: type,
        data: data,
        options: options,
    });
}
