"""
HTML report generation for backtest results.
"""

import json
from pathlib import Path

import pandas as pd

from risk.metrics import compute_all


REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Backtest Report — {strategy}</title>
<style>
body {{ font-family: 'Courier New', monospace; background: #0a0a0b; color: #e4e4e7; padding: 2rem; }}
h1 {{ color: #22d3ee; }}
h2 {{ color: #a1a1aa; border-bottom: 1px solid #27272a; padding-bottom: 0.5rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #27272a; }}
th {{ color: #22d3ee; }}
.positive {{ color: #34d399; }}
.negative {{ color: #ef4444; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
.metric-card {{ background: #111113; border: 1px solid #27272a; border-radius: 8px; padding: 1rem; }}
.metric-label {{ font-size: 0.8rem; color: #71717a; }}
.metric-value {{ font-size: 1.5rem; font-weight: bold; margin-top: 0.25rem; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p>Strategy: {strategy} | Ticker: {ticker} | Period: {period}</p>

<h2>Performance Metrics</h2>
<div class="metric-grid">
{metrics_html}
</div>

<h2>Trade Summary</h2>
<p>Total trades: {total_trades} | Win rate: {win_rate}%</p>

<h2>Recent Trades</h2>
<table>
<tr><th>Entry</th><th>Exit</th><th>Qty</th><th>PnL</th><th>Return</th></tr>
{trades_html}
</table>

</body>
</html>"""


def generate_report(
    strategy_name: str,
    ticker: str,
    equity_df: pd.DataFrame,
    trades: list[dict],
    output_path: str,
) -> str:
    """Generate an HTML backtest report."""
    returns = equity_df["equity"].pct_change().dropna()
    trades_pnl = [t["pnl"] for t in trades]
    metrics = compute_all(returns, trades_pnl)

    # Metrics cards
    metrics_html = ""
    for key, value in metrics.items():
        label = key.replace("_", " ").replace("pct", "%").title()
        css_class = ""
        if isinstance(value, (int, float)):
            css_class = "positive" if value > 0 else "negative"
        metrics_html += (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {css_class}">{value}</div>'
            f'</div>\n'
        )

    # Trades table (last 20)
    trades_html = ""
    for t in trades[-20:]:
        pnl_class = "positive" if t["pnl"] > 0 else "negative"
        trades_html += (
            f'<tr><td>{t.get("entry_price", "")}</td>'
            f'<td>{t.get("exit_price", "")}</td>'
            f'<td>{t.get("quantity", "")}</td>'
            f'<td class="{pnl_class}">${t["pnl"]:.2f}</td>'
            f'<td class="{pnl_class}">{t.get("return_pct", 0):.1f}%</td></tr>\n'
        )

    period = f"{equity_df['timestamp'].iloc[0]} → {equity_df['timestamp'].iloc[-1]}"

    html = REPORT_TEMPLATE.format(
        strategy=strategy_name,
        ticker=ticker,
        period=period,
        metrics_html=metrics_html,
        total_trades=len(trades),
        win_rate=metrics.get("win_rate_pct", 0),
        trades_html=trades_html,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path
