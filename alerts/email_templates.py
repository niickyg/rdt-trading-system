"""
Email Templates for RDT Trading System Alerts
Provides HTML and plain text templates for various alert types.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# Base HTML template with responsive design
BASE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* Reset and base styles */
        body, html {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}

        /* Container */
        .email-container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, {header_color} 0%, {header_color_dark} 100%);
            color: white;
            padding: 24px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }}

        .header .subtitle {{
            margin: 8px 0 0;
            opacity: 0.9;
            font-size: 14px;
        }}

        /* Priority badge */
        .priority-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            margin-top: 12px;
        }}

        .priority-low {{ background-color: rgba(255,255,255,0.2); }}
        .priority-normal {{ background-color: rgba(255,255,255,0.3); }}
        .priority-high {{ background-color: #ffc107; color: #333; }}
        .priority-critical {{ background-color: #dc3545; }}

        /* Content */
        .content {{
            padding: 24px;
        }}

        .content h2 {{
            color: #333;
            margin: 0 0 16px;
            font-size: 20px;
        }}

        .content p {{
            margin: 0 0 16px;
            color: #555;
        }}

        /* Info grid */
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}

        .info-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}

        .info-card .label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .info-card .value {{
            font-size: 20px;
            font-weight: 600;
            color: #333;
        }}

        .info-card .value.positive {{ color: #28a745; }}
        .info-card .value.negative {{ color: #dc3545; }}
        .info-card .value.neutral {{ color: #6c757d; }}

        /* Table styles */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .data-table th {{
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
        }}

        .data-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            color: #555;
        }}

        .data-table tr:last-child td {{
            border-bottom: none;
        }}

        /* Action button */
        .action-btn {{
            display: inline-block;
            background: linear-gradient(135deg, {header_color} 0%, {header_color_dark} 100%);
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 600;
            margin: 16px 0;
        }}

        /* Alert box */
        .alert-box {{
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
        }}

        .alert-box.warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}

        .alert-box.danger {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}

        .alert-box.success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }}

        .alert-box.info {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }}

        /* Footer */
        .footer {{
            background-color: #f8f9fa;
            padding: 20px 24px;
            text-align: center;
            border-top: 1px solid #dee2e6;
        }}

        .footer p {{
            margin: 0;
            font-size: 12px;
            color: #6c757d;
        }}

        .footer a {{
            color: {header_color};
            text-decoration: none;
        }}

        /* Responsive */
        @media only screen and (max-width: 480px) {{
            .header {{ padding: 16px; }}
            .header h1 {{ font-size: 20px; }}
            .content {{ padding: 16px; }}
            .info-grid {{ grid-template-columns: 1fr 1fr; gap: 12px; }}
            .info-card {{ padding: 12px; }}
            .info-card .value {{ font-size: 18px; }}
            .data-table th, .data-table td {{ padding: 8px; font-size: 14px; }}
        }}
    </style>
</head>
<body>
    <div style="padding: 20px; background-color: #f5f5f5;">
        <div class="email-container">
            {body}
        </div>
    </div>
</body>
</html>
"""


# Color schemes by priority
PRIORITY_COLORS = {
    'low': {'primary': '#6c757d', 'dark': '#5a6268'},
    'normal': {'primary': '#007bff', 'dark': '#0056b3'},
    'high': {'primary': '#fd7e14', 'dark': '#dc6a00'},
    'critical': {'primary': '#dc3545', 'dark': '#bd2130'},
}

# Color schemes by alert type
ALERT_TYPE_COLORS = {
    'trade': {'primary': '#28a745', 'dark': '#1e7e34'},
    'signal': {'primary': '#17a2b8', 'dark': '#117a8b'},
    'risk': {'primary': '#dc3545', 'dark': '#bd2130'},
    'system': {'primary': '#6610f2', 'dark': '#510bc4'},
    'summary': {'primary': '#007bff', 'dark': '#0056b3'},
}


@dataclass
class EmailContent:
    """Email content container."""
    subject: str
    html_body: str
    plain_text: str


def _get_timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _format_currency(value: float) -> str:
    """Format number as currency."""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _format_percent(value: float) -> str:
    """Format number as percentage."""
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.2f}%"


def _get_value_class(value: float) -> str:
    """Get CSS class for positive/negative value."""
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    return 'neutral'


class EmailTemplates:
    """Email template generator for trading alerts."""

    @staticmethod
    def trade_alert(
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        reason: str,
        strategy: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        priority: str = 'normal',
        timestamp: Optional[str] = None
    ) -> EmailContent:
        """
        Generate trade alert email content.

        Args:
            action: Trade action (BUY, SELL, STOP_LOSS, TAKE_PROFIT)
            symbol: Stock symbol
            price: Trade price
            quantity: Number of shares
            reason: Reason for the trade
            strategy: Strategy name (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            priority: Alert priority
            timestamp: Alert timestamp (optional)

        Returns:
            EmailContent: Generated email content
        """
        timestamp = timestamp or _get_timestamp()
        total_value = price * quantity

        # Determine colors based on action
        action_upper = action.upper()
        if action_upper in ('BUY', 'ENTRY'):
            colors = {'primary': '#28a745', 'dark': '#1e7e34'}
            action_icon = 'LONG'
        elif action_upper in ('SELL', 'EXIT', 'STOP_LOSS', 'TAKE_PROFIT'):
            colors = {'primary': '#dc3545', 'dark': '#bd2130'}
            action_icon = 'EXIT' if action_upper != 'STOP_LOSS' else 'STOP'
        else:
            colors = ALERT_TYPE_COLORS['trade']
            action_icon = action_upper[:4]

        priority_class = f"priority-{priority.lower()}"

        # Build HTML body
        html_body = f"""
            <div class="header">
                <h1>{action_icon}: {symbol}</h1>
                <div class="subtitle">{timestamp}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <h2>Trade Executed</h2>
                <p>{reason}</p>

                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">Action</div>
                        <div class="value">{action_upper}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Symbol</div>
                        <div class="value">{symbol}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Price</div>
                        <div class="value">{_format_currency(price)}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Quantity</div>
                        <div class="value">{quantity}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Total Value</div>
                        <div class="value">{_format_currency(total_value)}</div>
                    </div>
        """

        if strategy:
            html_body += f"""
                    <div class="info-card">
                        <div class="label">Strategy</div>
                        <div class="value" style="font-size: 14px;">{strategy}</div>
                    </div>
            """

        html_body += "</div>"

        # Add stop loss and take profit if provided
        if stop_loss or take_profit:
            html_body += '<table class="data-table"><thead><tr><th>Level</th><th>Price</th></tr></thead><tbody>'
            if stop_loss:
                html_body += f'<tr><td>Stop Loss</td><td class="negative">{_format_currency(stop_loss)}</td></tr>'
            if take_profit:
                html_body += f'<tr><td>Take Profit</td><td class="positive">{_format_currency(take_profit)}</td></tr>'
            html_body += '</tbody></table>'

        html_body += """
            </div>

            <div class="footer">
                <p>RDT Trading System | <a href="#">View Dashboard</a></p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=f"Trade Alert: {action_upper} {symbol}",
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        # Plain text version
        plain_text = f"""
RDT TRADING ALERT - {action_upper}: {symbol}
{'=' * 50}

Time: {timestamp}
Priority: {priority.upper()}

Trade Details:
- Action: {action_upper}
- Symbol: {symbol}
- Price: {_format_currency(price)}
- Quantity: {quantity}
- Total Value: {_format_currency(total_value)}
{f'- Strategy: {strategy}' if strategy else ''}

Reason: {reason}

{f'Stop Loss: {_format_currency(stop_loss)}' if stop_loss else ''}
{f'Take Profit: {_format_currency(take_profit)}' if take_profit else ''}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[Trade Alert] {action_upper} {quantity} {symbol} @ {_format_currency(price)}",
            html_body=full_html,
            plain_text=plain_text
        )

    @staticmethod
    def signal_alert(
        signal_type: str,
        symbol: str,
        price: float,
        indicator: str,
        indicator_value: float,
        recommendation: str,
        confidence: Optional[float] = None,
        priority: str = 'normal',
        timestamp: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> EmailContent:
        """
        Generate signal alert email content.

        Args:
            signal_type: Type of signal (BULLISH, BEARISH, BREAKOUT, etc.)
            symbol: Stock symbol
            price: Current price
            indicator: Indicator name (RRS, RSI, MACD, etc.)
            indicator_value: Current indicator value
            recommendation: Action recommendation
            confidence: Confidence level 0-100 (optional)
            priority: Alert priority
            timestamp: Alert timestamp (optional)
            additional_data: Additional indicator data (optional)

        Returns:
            EmailContent: Generated email content
        """
        timestamp = timestamp or _get_timestamp()
        colors = ALERT_TYPE_COLORS['signal']
        priority_class = f"priority-{priority.lower()}"

        # Determine signal direction
        signal_upper = signal_type.upper()
        if 'BULLISH' in signal_upper or 'BUY' in signal_upper or 'BREAKOUT' in signal_upper:
            signal_class = 'positive'
            signal_icon = 'UP'
        elif 'BEARISH' in signal_upper or 'SELL' in signal_upper or 'BREAKDOWN' in signal_upper:
            signal_class = 'negative'
            signal_icon = 'DN'
        else:
            signal_class = 'neutral'
            signal_icon = '--'

        html_body = f"""
            <div class="header">
                <h1>Signal: {symbol}</h1>
                <div class="subtitle">{timestamp}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <h2>{signal_type}</h2>
                <p>{recommendation}</p>

                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">Symbol</div>
                        <div class="value">{symbol}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Price</div>
                        <div class="value">{_format_currency(price)}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">{indicator}</div>
                        <div class="value {signal_class}">{indicator_value:.2f}</div>
                    </div>
        """

        if confidence is not None:
            html_body += f"""
                    <div class="info-card">
                        <div class="label">Confidence</div>
                        <div class="value">{confidence:.0f}%</div>
                    </div>
            """

        html_body += "</div>"

        # Add additional data if provided
        if additional_data:
            html_body += '<table class="data-table"><thead><tr><th>Indicator</th><th>Value</th></tr></thead><tbody>'
            for key, value in additional_data.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                html_body += f'<tr><td>{key}</td><td>{formatted_value}</td></tr>'
            html_body += '</tbody></table>'

        html_body += f"""
                <div class="alert-box info">
                    <strong>Recommendation:</strong> {recommendation}
                </div>
            </div>

            <div class="footer">
                <p>RDT Trading System | <a href="#">View Chart</a></p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=f"Signal Alert: {symbol} - {signal_type}",
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        # Plain text version
        plain_text = f"""
RDT SIGNAL ALERT - {symbol}
{'=' * 50}

Time: {timestamp}
Priority: {priority.upper()}
Signal: {signal_type}

Details:
- Symbol: {symbol}
- Price: {_format_currency(price)}
- {indicator}: {indicator_value:.2f}
{f'- Confidence: {confidence:.0f}%' if confidence else ''}

Recommendation: {recommendation}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[Signal] {signal_type} - {symbol} @ {_format_currency(price)}",
            html_body=full_html,
            plain_text=plain_text
        )

    @staticmethod
    def daily_summary(
        date: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        total_pnl_percent: float,
        portfolio_value: float,
        starting_value: float,
        trades: Optional[List[Dict[str, Any]]] = None,
        top_winners: Optional[List[Dict[str, Any]]] = None,
        top_losers: Optional[List[Dict[str, Any]]] = None,
        signals_generated: int = 0,
        priority: str = 'normal'
    ) -> EmailContent:
        """
        Generate daily performance summary email.

        Args:
            date: Summary date
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss in dollars
            total_pnl_percent: Total profit/loss in percent
            portfolio_value: Current portfolio value
            starting_value: Portfolio value at start of day
            trades: List of trade details (optional)
            top_winners: List of top winning trades (optional)
            top_losers: List of top losing trades (optional)
            signals_generated: Number of signals generated
            priority: Alert priority

        Returns:
            EmailContent: Generated email content
        """
        colors = ALERT_TYPE_COLORS['summary']
        priority_class = f"priority-{priority.lower()}"

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_class = _get_value_class(total_pnl)
        day_change = portfolio_value - starting_value

        html_body = f"""
            <div class="header">
                <h1>Daily Summary</h1>
                <div class="subtitle">{date}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <h2>Performance Overview</h2>

                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">P&L</div>
                        <div class="value {pnl_class}">{_format_currency(total_pnl)}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Return</div>
                        <div class="value {pnl_class}">{_format_percent(total_pnl_percent)}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Portfolio</div>
                        <div class="value">{_format_currency(portfolio_value)}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Day Change</div>
                        <div class="value {_get_value_class(day_change)}">{_format_currency(day_change)}</div>
                    </div>
                </div>

                <h2>Trading Activity</h2>

                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">Total Trades</div>
                        <div class="value">{total_trades}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Winners</div>
                        <div class="value positive">{winning_trades}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Losers</div>
                        <div class="value negative">{losing_trades}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Win Rate</div>
                        <div class="value">{win_rate:.1f}%</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Signals</div>
                        <div class="value">{signals_generated}</div>
                    </div>
                </div>
        """

        # Add top winners
        if top_winners:
            html_body += """
                <h2>Top Winners</h2>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>P&L</th>
                            <th>Return</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for trade in top_winners[:5]:
                html_body += f"""
                        <tr>
                            <td>{trade.get('symbol', 'N/A')}</td>
                            <td class="positive">{_format_currency(trade.get('pnl', 0))}</td>
                            <td class="positive">{_format_percent(trade.get('return_pct', 0))}</td>
                        </tr>
                """
            html_body += "</tbody></table>"

        # Add top losers
        if top_losers:
            html_body += """
                <h2>Top Losers</h2>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>P&L</th>
                            <th>Return</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for trade in top_losers[:5]:
                html_body += f"""
                        <tr>
                            <td>{trade.get('symbol', 'N/A')}</td>
                            <td class="negative">{_format_currency(trade.get('pnl', 0))}</td>
                            <td class="negative">{_format_percent(trade.get('return_pct', 0))}</td>
                        </tr>
                """
            html_body += "</tbody></table>"

        html_body += """
            </div>

            <div class="footer">
                <p>RDT Trading System | <a href="#">View Full Report</a></p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=f"Daily Summary - {date}",
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        # Plain text version
        plain_text = f"""
RDT DAILY SUMMARY - {date}
{'=' * 50}

PERFORMANCE OVERVIEW
--------------------
P&L: {_format_currency(total_pnl)} ({_format_percent(total_pnl_percent)})
Portfolio Value: {_format_currency(portfolio_value)}
Day Change: {_format_currency(day_change)}

TRADING ACTIVITY
----------------
Total Trades: {total_trades}
Winners: {winning_trades}
Losers: {losing_trades}
Win Rate: {win_rate:.1f}%
Signals Generated: {signals_generated}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[Daily Summary] {date} | P&L: {_format_currency(total_pnl)} ({_format_percent(total_pnl_percent)})",
            html_body=full_html,
            plain_text=plain_text
        )

    @staticmethod
    def risk_alert(
        alert_type: str,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None,
        recommendation: Optional[str] = None,
        priority: str = 'high',
        timestamp: Optional[str] = None
    ) -> EmailContent:
        """
        Generate risk alert email content.

        Args:
            alert_type: Type of risk alert (DRAWDOWN, POSITION_SIZE, DAILY_LOSS, etc.)
            message: Alert message
            current_value: Current metric value
            threshold_value: Threshold that was breached
            symbol: Related symbol (optional)
            recommendation: Recommended action (optional)
            priority: Alert priority
            timestamp: Alert timestamp (optional)

        Returns:
            EmailContent: Generated email content
        """
        timestamp = timestamp or _get_timestamp()
        colors = ALERT_TYPE_COLORS['risk']
        priority_class = f"priority-{priority.lower()}"

        # Determine alert box class based on priority
        alert_box_class = 'danger' if priority.lower() in ('high', 'critical') else 'warning'

        html_body = f"""
            <div class="header">
                <h1>Risk Alert</h1>
                <div class="subtitle">{timestamp}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <div class="alert-box {alert_box_class}">
                    <strong>{alert_type}</strong><br>
                    {message}
                </div>

                <div class="info-grid">
        """

        if symbol:
            html_body += f"""
                    <div class="info-card">
                        <div class="label">Symbol</div>
                        <div class="value">{symbol}</div>
                    </div>
            """

        html_body += f"""
                    <div class="info-card">
                        <div class="label">Current Value</div>
                        <div class="value negative">{current_value:.2f}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Threshold</div>
                        <div class="value">{threshold_value:.2f}</div>
                    </div>
                </div>
        """

        if recommendation:
            html_body += f"""
                <div class="alert-box info">
                    <strong>Recommended Action:</strong> {recommendation}
                </div>
            """

        html_body += """
            </div>

            <div class="footer">
                <p>RDT Trading System | <a href="#">View Risk Dashboard</a></p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=f"Risk Alert: {alert_type}",
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        # Plain text version
        symbol_line = f"Symbol: {symbol}\n" if symbol else ""
        recommendation_line = f"\nRecommendation: {recommendation}" if recommendation else ""

        plain_text = f"""
RDT RISK ALERT - {alert_type}
{'=' * 50}

Time: {timestamp}
Priority: {priority.upper()}

{message}

{symbol_line}Current Value: {current_value:.2f}
Threshold: {threshold_value:.2f}
{recommendation_line}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[RISK ALERT] {alert_type}" + (f" - {symbol}" if symbol else ""),
            html_body=full_html,
            plain_text=plain_text
        )

    @staticmethod
    def system_alert(
        title: str,
        message: str,
        alert_type: str = 'INFO',
        details: Optional[Dict[str, Any]] = None,
        priority: str = 'normal',
        timestamp: Optional[str] = None
    ) -> EmailContent:
        """
        Generate system alert email content.

        Args:
            title: Alert title
            message: Alert message
            alert_type: Type of system alert (INFO, WARNING, ERROR, etc.)
            details: Additional details (optional)
            priority: Alert priority
            timestamp: Alert timestamp (optional)

        Returns:
            EmailContent: Generated email content
        """
        timestamp = timestamp or _get_timestamp()
        colors = ALERT_TYPE_COLORS['system']
        priority_class = f"priority-{priority.lower()}"

        # Determine alert box class based on type
        type_upper = alert_type.upper()
        if type_upper in ('ERROR', 'CRITICAL'):
            alert_box_class = 'danger'
        elif type_upper == 'WARNING':
            alert_box_class = 'warning'
        elif type_upper == 'SUCCESS':
            alert_box_class = 'success'
        else:
            alert_box_class = 'info'

        html_body = f"""
            <div class="header">
                <h1>System Alert</h1>
                <div class="subtitle">{timestamp}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <h2>{title}</h2>

                <div class="alert-box {alert_box_class}">
                    <strong>{alert_type}:</strong> {message}
                </div>
        """

        if details:
            html_body += """
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for key, value in details.items():
                html_body += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                """
            html_body += "</tbody></table>"

        html_body += """
            </div>

            <div class="footer">
                <p>RDT Trading System | <a href="#">View System Status</a></p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=f"System Alert: {title}",
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        # Plain text version
        details_text = ""
        if details:
            details_text = "\nDetails:\n" + "\n".join(f"- {k}: {v}" for k, v in details.items())

        plain_text = f"""
RDT SYSTEM ALERT - {title}
{'=' * 50}

Time: {timestamp}
Priority: {priority.upper()}
Type: {alert_type}

{message}
{details_text}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[System] {alert_type}: {title}",
            html_body=full_html,
            plain_text=plain_text
        )

    @staticmethod
    def generic_alert(
        title: str,
        message: str,
        priority: str = 'normal',
        timestamp: Optional[str] = None
    ) -> EmailContent:
        """
        Generate generic alert email content.

        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            timestamp: Alert timestamp (optional)

        Returns:
            EmailContent: Generated email content
        """
        timestamp = timestamp or _get_timestamp()
        colors = PRIORITY_COLORS.get(priority.lower(), PRIORITY_COLORS['normal'])
        priority_class = f"priority-{priority.lower()}"

        html_body = f"""
            <div class="header">
                <h1>{title}</h1>
                <div class="subtitle">{timestamp}</div>
                <span class="priority-badge {priority_class}">{priority.upper()}</span>
            </div>

            <div class="content">
                <p>{message}</p>
            </div>

            <div class="footer">
                <p>RDT Trading System</p>
            </div>
        """

        full_html = BASE_HTML_TEMPLATE.format(
            title=title,
            header_color=colors['primary'],
            header_color_dark=colors['dark'],
            body=html_body
        )

        plain_text = f"""
RDT ALERT - {title}
{'=' * 50}

Time: {timestamp}
Priority: {priority.upper()}

{message}

--
RDT Trading System
        """.strip()

        return EmailContent(
            subject=f"[Alert] {title}",
            html_body=full_html,
            plain_text=plain_text
        )
