# -*- coding: utf-8 -*-
"""
Stock Portfolio Tracker (Tkinter GUI)
Upgrades:
- Selective Edit dialog (rename/quantity/buy price; blanks keep old)
- Loader overlay closable (only overlay closes; task continues)
- Two charts in separate windows (non-blocking)
- Faster UI feel: buttons disable during work, batched price fetch
- Nicer theme & colors: green/red P&L, zebra rows, polished ttk
- Keyboard shortcuts: Ctrl+S, Ctrl+O, Ctrl+Q, F5, Insert, Delete, Ctrl+E
- Clear dependency/guard messages (e.g., load before save)
- Clean exits from GUI (Exit, ✖️) and terminal (Ctrl+C)
"""

import os
import threading
from datetime import datetime

# --- Data / Math ---
import yfinance as yf
import pandas as pd
import numpy as np

# --- Plotting (Tkinter-friendly backend) ---
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# --- GUI ---
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import simpledialog


# ==============================
# Global / Storage
# ==============================
portfolio = {
    "AAPL": {"quantity": 10, "buy_price": 150},
    "MSFT": {"quantity": 5, "buy_price": 300},
    "GOOGL": {"quantity": 2, "buy_price": 2500},
}
TRANSACTION_FILE = "transaction_history.csv"


# ==============================
# Utility: Modal Loading Overlay
# ==============================
class Loader:
    """
    A small modal 'working...' dialog shown during long tasks.
    NOTE: You can close it with ✖️; task will still continue in background.
    """

    def __init__(self, parent, text="Abhi kaam ho raha hai (Loading) … ⏳"):
        self.top = tk.Toplevel(parent)
        self.top.title("Please wait")
        self.top.geometry("340x120")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()  # modal

        # Center relative to parent
        self.top.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 170
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 60
        self.top.geometry(f"+{x}+{y}")

        ttk.Label(self.top, text=text, font=("Segoe UI", 11)).pack(pady=(14, 6))
        self.pb = ttk.Progressbar(self.top, mode="indeterminate", length=280)
        self.pb.pack(pady=(0, 8))
        self.pb.start(10)

        # Allow closing overlay (does NOT cancel the background job)
        self.top.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        try:
            self.pb.stop()
        except Exception:
            pass
        try:
            self.top.grab_release()
        except Exception:
            pass
        self.top.destroy()


def _set_buttons_state(state: str):
    """Enable/disable all ttk.Buttons so app feels snappy and avoids double-clicks."""
    for child in button_frame.winfo_children():
        if isinstance(child, ttk.Button):
            child.configure(state=state)


def run_with_loader(fn):
    """
    Decorator: runs a function in a background thread while showing loader.
    - Disables buttons for the duration
    - Shows a closable overlay immediately
    - Keeps UI responsive
    """

    def wrapper(*args, **kwargs):
        loader = Loader(root)
        _set_buttons_state("disabled")

        def task():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                # Show error safely on the main thread
                def show_err():
                    messagebox.showerror("Error", f"Unexpected error:\n{e}")

                root.after(0, show_err)
            finally:
                # Close loader & re-enable buttons on the main thread
                def finish():
                    try:
                        loader.close()
                    except Exception:
                        pass
                    _set_buttons_state("normal")

                root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    return wrapper


# ==============================
# Helpers
# ==============================
def set_status(text):
    """Update bottom status bar text."""
    status_var.set(text)
    root.update_idletasks()


def log_transaction(action, ticker, quantity=None, buy_price=None, note=None):
    """Append a transaction row into CSV (create if missing)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": timestamp,
        "Action": action,
        "Ticker": ticker,
        "Quantity": quantity,
        "Buy Price": buy_price,
        "Note": note or "",
    }
    df = pd.DataFrame([row])
    if os.path.exists(TRANSACTION_FILE):
        df.to_csv(TRANSACTION_FILE, mode="a", index=False, header=False)
    else:
        df.to_csv(TRANSACTION_FILE, index=False)


# ==============================
# Data Fetch (Fast path)
# ==============================
def batch_latest_prices(tickers):
    """
    Fetch latest close for multiple tickers in a single request (fast).
    Returns dict: {ticker: price or None}
    """
    prices = {t: None for t in tickers}
    if not tickers:
        return prices

    try:
        data = yf.download(
            tickers=tickers,
            period="1d",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )
    except Exception:
        return prices

    # If single ticker, columns may be simple not multiindex
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                close_series = data[t]["Close"].dropna()
                if not close_series.empty:
                    prices[t] = float(close_series.iloc[-1])
            except Exception:
                prices[t] = None
    else:
        # Single ticker case
        try:
            close_series = data["Close"].dropna()
            if not close_series.empty:
                prices[tickers[0]] = float(close_series.iloc[-1])
        except Exception:
            pass

    return prices


# ==============================
# Core Portfolio Calculations
# ==============================
def calculate_portfolio():
    """
    Build a DataFrame with latest prices + P/L absolute and percent + weights.
    Uses batch download for speed.
    """
    tickers = list(portfolio.keys())
    latest = batch_latest_prices(tickers)

    rows = []
    for ticker, details in portfolio.items():
        current_price = latest.get(ticker)
        if current_price is None:
            # skip if no valid price
            continue

        quantity = float(details["quantity"])
        buy_price = float(details["buy_price"])

        investment = buy_price * quantity
        current_value = current_price * quantity
        profit_loss = current_value - investment
        pl_pct = (profit_loss / investment * 100.0) if investment > 0 else 0.0

        rows.append(
            {
                "Ticker": ticker,
                "Quantity": int(quantity),
                "Buy Price": round(buy_price, 2),
                "Current Price": round(current_price, 2),
                "Investment ($)": round(investment, 2),
                "Current Value ($)": round(current_value, 2),
                "Profit/Loss ($)": round(profit_loss, 2),
                "P/L %": round(pl_pct, 2),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        total_val = df["Current Value ($)"].sum()
        df["Weight %"] = (
            ((df["Current Value ($)"] / total_val) * 100.0).round(2)
            if total_val > 0
            else 0.0
        )
    return df


# ==============================
# Risk & Analytics
# ==============================
def get_price_history_for_portfolio(period="1y", interval="1d"):
    """
    Download adjusted close price history for all tickers and S&P500 (^GSPC).
    Returns (prices_df, bench_series). prices_df columns = tickers.
    """
    tickers = list(portfolio.keys())
    if not tickers:
        return pd.DataFrame(), pd.Series(dtype=float)

    try:
        prices = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame(), pd.Series(dtype=float)

    def get_close(df):
        if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
            return df["Adj Close"]
        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            return df["Close"]
        if isinstance(df, pd.Series):
            return df
        return pd.Series(dtype=float)

    wide = pd.DataFrame()
    if isinstance(prices.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr in prices.columns.get_level_values(0):
                series = get_close(prices[tkr])
                if not series.empty:
                    wide[tkr] = series
    else:
        series = get_close(prices)
        if not series.empty:
            wide[tickers[0]] = series

    bench = yf.download(
        "^GSPC", period=period, interval=interval, auto_adjust=True, progress=False
    )
    bench_close = (
        bench["Adj Close"]
        if ("Adj Close" in bench.columns and not bench.empty)
        else pd.Series(dtype=float)
    )

    if not wide.empty and not bench_close.empty:
        wide, bench_close = wide.align(bench_close, join="inner", axis=0)

    return wide.dropna(how="all"), bench_close.dropna()


def compute_portfolio_returns(prices_df):
    """Value-weighted portfolio daily returns from historical prices."""
    if prices_df.empty:
        return pd.Series(dtype=float)

    current_df = calculate_portfolio()
    if current_df.empty:
        return pd.Series(dtype=float)

    total_val = current_df["Current Value ($)"].sum()
    weights = {
        r["Ticker"]: (r["Current Value ($)"] / total_val if total_val > 0 else 0.0)
        for _, r in current_df.iterrows()
    }

    rets = prices_df.pct_change().dropna()
    w_vec = np.array([weights.get(t, 0.0) for t in rets.columns])
    port_rets = (rets * w_vec).sum(axis=1)
    return port_rets


def compute_risk_metrics(period="1y", interval="1d"):
    """
    Returns dict with:
      expected_return (annualized), volatility (annualized), beta (vs ^GSPC)
    """
    prices_df, bench = get_price_history_for_portfolio(period=period, interval=interval)
    if prices_df.empty or bench.empty:
        return {"expected_return": None, "volatility": None, "beta": None, "n_days": 0}

    port_rets = compute_portfolio_returns(prices_df)
    bench_rets = bench.pct_change().dropna()
    port_rets, bench_rets = port_rets.align(bench_rets, join="inner")

    if port_rets.empty or bench_rets.empty:
        return {"expected_return": None, "volatility": None, "beta": None, "n_days": 0}

    mu = port_rets.mean()
    sigma = port_rets.std(ddof=1)
    cov = np.cov(port_rets, bench_rets)[0, 1]
    bench_var = bench_rets.var(ddof=1)

    exp_ret_annual = mu * 252
    vol_annual = sigma * np.sqrt(252)
    beta = cov / bench_var if bench_var > 0 else None

    return {
        "expected_return": float(exp_ret_annual),
        "volatility": float(vol_annual),
        "beta": float(beta) if beta is not None else None,
        "n_days": int(len(port_rets)),
    }


# ==============================
# GUI: Table refresh (with row colors)
# ==============================
def refresh_tree(df=None):
    """
    Refresh the main table (Treeview) with a DataFrame.
    - Rows with profit in green, loss in red
    - Zebra striping for readability
    """
    if df is None:
        df = calculate_portfolio()

    for i in tree.get_children():
        tree.delete(i)

    if df.empty:
        return

    for idx, (_, row) in enumerate(df.iterrows()):
        tag = "profit" if row["Profit/Loss ($)"] >= 0 else "loss"
        zebra = "even" if idx % 2 == 0 else "odd"
        tree.insert(
            "",
            "end",
            values=(
                row["Ticker"],
                row["Quantity"],
                row["Buy Price"],
                row["Current Price"],
                row["Investment ($)"],
                row["Current Value ($)"],
                row["Profit/Loss ($)"],
                f'{row["P/L %"]}%',
                f'{row.get("Weight %", 0.0)}%',
            ),
            tags=(tag, zebra),
        )


# ==============================
# Dialogs (Add/Edit with defaults)
# ==============================
def ask_add():
    """
    Add stock dialog:
    - Ticker (required)
    - Quantity (required)
    - Buy Price (required)
    """
    t = simple_input("Add Stock", "Enter Stock Ticker (e.g., AAPL):")
    if not t:
        return
    try:
        q = int(simple_input("Add Stock", f"Quantity for {t.upper()}:", "10"))
        p = float(simple_input("Add Stock", f"Buy Price for {t.upper()}:", "100"))
    except Exception:
        messagebox.showerror("Error", "Invalid input ❌")
        return
    portfolio[t.upper()] = {"quantity": q, "buy_price": p}
    log_transaction("Add", t.upper(), q, p)
    messagebox.showinfo("Success", f"{t.upper()} added ✅")
    view_portfolio()


def ask_edit():
    """
    Selective edit dialog:
    - Pre-fills current values
    - Any field left blank will keep the old value
    - You can also rename the ticker
    """
    sel = tree.selection()
    if not sel:
        messagebox.showwarning("Warning", "Select a stock to edit ⚠")
        return
    item = tree.item(sel[0])
    old_ticker = item["values"][0]
    current = portfolio.get(old_ticker, {})
    if not current:
        messagebox.showerror("Error", "Could not find selected ticker ❌")
        return

    # Input boxes with defaults; blank keeps old
    new_ticker = simple_input(
        "Edit Stock",
        f"New Ticker for {old_ticker} (blank = keep):",
        old_ticker,
        allow_blank=True,
    )
    qty_str = simple_input(
        "Edit Stock",
        f"New Quantity for {old_ticker} (blank = keep {current['quantity']}):",
        str(current["quantity"]),
        allow_blank=True,
    )
    price_str = simple_input(
        "Edit Stock",
        f"New Buy Price for {old_ticker} (blank = keep {current['buy_price']}):",
        str(current["buy_price"]),
        allow_blank=True,
    )

    # Resolve values
    final_ticker = (new_ticker or old_ticker).upper()
    try:
        final_qty = int(qty_str) if qty_str.strip() != "" else current["quantity"]
        final_price = (
            float(price_str) if price_str.strip() != "" else current["buy_price"]
        )
    except Exception:
        messagebox.showerror("Error", "Invalid quantity/price ❌")
        return

    # Apply (rename key if needed)
    note = None
    if final_ticker != old_ticker:
        # Prevent overwrite accidental
        if final_ticker in portfolio and final_ticker != old_ticker:
            messagebox.showerror("Error", f"{final_ticker} already exists ❌")
            return
        portfolio.pop(old_ticker, None)
        note = f"Renamed from {old_ticker}"
    portfolio[final_ticker] = {"quantity": final_qty, "buy_price": final_price}

    log_transaction("Edit", final_ticker, final_qty, final_price, note=note)
    messagebox.showinfo("Success", f"{final_ticker} updated ✅")
    view_portfolio()


def simple_input(title, prompt, initial="", allow_blank=False):
    """
    Wrapper around simpledialog.askstring with initial value
    and optional blank allowance.
    """
    val = simpledialog.askstring(title, prompt, initialvalue=initial)
    if val is None:
        return "" if allow_blank else None
    if not allow_blank and val.strip() == "":
        return None
    return val


def ask_delete():
    sel = tree.selection()
    if not sel:
        messagebox.showwarning("Warning", "Select a stock to delete ⚠")
        return
    item = tree.item(sel[0])
    ticker = item["values"][0]
    if messagebox.askyesno("Confirm Delete", f"Delete {ticker} from portfolio?"):
        portfolio.pop(ticker, None)
        log_transaction("Delete", ticker)
        messagebox.showinfo("Success", f"{ticker} deleted ✅")
        view_portfolio()


def view_transactions():
    if not os.path.exists(TRANSACTION_FILE):
        messagebox.showwarning(
            "Warning",
            "No transactions yet ⚠\nTip: Add/Edit/Delete a stock to create history.",
        )
        return
    df = pd.read_csv(TRANSACTION_FILE)
    top = tk.Toplevel()
    top.title("Transaction History")
    tree_tx = ttk.Treeview(top, columns=df.columns.tolist(), show="headings")
    for col in df.columns:
        tree_tx.heading(col, text=col)
        tree_tx.column(col, width=140)
    tree_tx.pack(fill="both", expand=True)
    for _, r in df.iterrows():
        tree_tx.insert("", "end", values=list(r))


# ==============================
# GUI Actions (wrapped with loader for responsiveness)
# ==============================
@run_with_loader
def view_portfolio():
    set_status("Fetching latest prices…")
    df = calculate_portfolio()
    if df.empty:
        messagebox.showwarning("Warning", "No valid data found ⚠")
    refresh_tree(df)
    set_status("Ready ✅")


@run_with_loader
def generate_charts_gui():
    """
    Show Profit/Loss bar and Allocation pie in two SEPARATE Tkinter windows.
    Each chart is closable with the window's ❌ button.
    """
    set_status("Generating charts…")
    df = calculate_portfolio()
    if df.empty:
        messagebox.showwarning("Warning", "Portfolio empty ⚠")
        set_status("Ready")
        return

    sns.set_style("darkgrid")

    # --- Bar Chart Window ---
    win_bar = tk.Toplevel(root)
    win_bar.title("Profit / Loss per Stock")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="Ticker",
        y="Profit/Loss ($)",
        data=df,
        hue="Ticker",
        dodge=False,
        legend=False,
        palette="coolwarm",
        ax=ax1,
    )
    ax1.set_title("Profit / Loss per Stock", fontsize=14, fontweight="bold")
    ax1.axhline(0, linewidth=1)
    fig1.tight_layout()

    canvas1 = FigureCanvasTkAgg(fig1, master=win_bar)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True)

    # --- Pie Chart Window ---
    win_pie = tk.Toplevel(root)
    win_pie.title("Portfolio Allocation by Value")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.pie(
        df["Current Value ($)"],
        labels=df["Ticker"],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    ax2.set_title("Portfolio Allocation by Value", fontsize=14, fontweight="bold")
    fig2.tight_layout()

    canvas2 = FigureCanvasTkAgg(fig2, master=win_pie)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True)

    set_status("Ready ✅")


@run_with_loader
def show_gainers_losers():
    set_status("Computing gainers/losers…")
    df = calculate_portfolio()
    if df.empty:
        messagebox.showwarning("Warning", "Portfolio empty ⚠")
        set_status("Ready")
        return
    df_sorted = df.sort_values(by="P/L %", ascending=False)
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5)

    top = tk.Toplevel()
    top.title("Top Gainers / Losers (by P/L %)")
    ttk.Label(top, text="Top Gainers", font=("Segoe UI", 11, "bold")).pack(pady=(10, 2))
    tree_top = ttk.Treeview(
        top, columns=top5.columns.tolist(), show="headings", height=6
    )
    for c in top5.columns:
        tree_top.heading(c, text=c)
        tree_top.column(c, width=110)
    tree_top.pack(fill="x", padx=10)
    for _, r in top5.iterrows():
        tree_top.insert("", "end", values=list(r))

    ttk.Label(top, text="Top Losers", font=("Segoe UI", 11, "bold")).pack(pady=(10, 2))
    tree_bottom = ttk.Treeview(
        top, columns=bottom5.columns.tolist(), show="headings", height=6
    )
    for c in bottom5.columns:
        tree_bottom.heading(c, text=c)
        tree_bottom.column(c, width=110)
    tree_bottom.pack(fill="x", padx=10, pady=(0, 10))
    for _, r in bottom5.iterrows():
        tree_bottom.insert("", "end", values=list(r))

    set_status("Ready ✅")


@run_with_loader
def show_allocation_insights():
    set_status("Analyzing allocation…")
    df = calculate_portfolio()
    if df.empty:
        messagebox.showwarning("Warning", "Portfolio empty ⚠")
        set_status("Ready")
        return

    msgs = []
    threshold_major = 40.0  # ≥ 40% is over-concentrated
    threshold_high = 25.0  # highlight ≥ 25%

    heavy = df[df["Weight %"] >= threshold_major]
    high = df[(df["Weight %"] >= threshold_high) & (df["Weight %"] < threshold_major)]

    if not heavy.empty:
        tickers = ", ".join(
            f'{r["Ticker"]} ({r["Weight %"]}%)' for _, r in heavy.iterrows()
        )
        msgs.append(
            f"⚠ Over-concentration: {tickers} ≥ {threshold_major}% of portfolio."
        )
    if not high.empty:
        tickers = ", ".join(
            f'{r["Ticker"]} ({r["Weight %"]}%)' for _, r in high.iterrows()
        )
        msgs.append(
            f"Note: High allocation positions: {tickers} (≥ {threshold_high}%)."
        )
    if not msgs:
        msgs.append("✅ Allocation looks reasonably diversified.")

    messagebox.showinfo("Allocation Insights", "\n".join(msgs))
    set_status("Ready ✅")


@run_with_loader
def show_risk_metrics():
    set_status("Fetching 1Y history for risk metrics…")
    metrics = compute_risk_metrics(period="1y", interval="1d")
    if metrics["n_days"] == 0 or any(
        v is None
        for v in [metrics["expected_return"], metrics["volatility"], metrics["beta"]]
    ):
        messagebox.showwarning(
            "Warning", "Could not compute risk metrics (insufficient data) ⚠"
        )
        set_status("Ready")
        return

    exp_ret = metrics["expected_return"] * 100.0
    vol = metrics["volatility"] * 100.0
    beta = metrics["beta"]

    text = (
        f"📈 Portfolio Risk Metrics (≈{metrics['n_days']} trading days)\n\n"
        f"• Expected Return (annualized): {exp_ret:.2f}%\n"
        f"• Volatility (annualized): {vol:.2f}%\n"
        f"• Beta vs S&P 500: {beta:.2f}\n\n"
        f"Note: Estimates are based on historical data; not guarantees."
    )
    messagebox.showinfo("Risk Metrics", text)
    set_status("Ready ✅")


@run_with_loader
def save_portfolio_gui():
    set_status("Saving portfolio…")
    df = calculate_portfolio()
    if df.empty:
        messagebox.showwarning(
            "Warning",
            "Nothing to save ⚠\nTip: Add stocks or refresh View Portfolio first.",
        )
        set_status("Ready")
        return
    df.to_csv("portfolio_report.csv", index=False)
    df.to_excel("portfolio_report.xlsx", index=False)
    messagebox.showinfo("Success", "Portfolio saved to CSV & Excel ✅")
    set_status("Ready ✅")


@run_with_loader
def load_portfolio_gui():
    set_status("Loading portfolio from CSV…")
    global portfolio
    if os.path.exists("portfolio_report.csv"):
        df = pd.read_csv("portfolio_report.csv")
        if df.empty:
            messagebox.showwarning("Warning", "CSV is empty ⚠")
            set_status("Ready")
            return
        new_map = {}
        for _, r in df.iterrows():
            new_map[str(r["Ticker"]).upper()] = {
                "quantity": int(r["Quantity"]),
                "buy_price": float(r["Buy Price"]),
            }
        portfolio = new_map
        messagebox.showinfo("Success", "Portfolio loaded from CSV ✅")
        refresh_tree(df)
    else:
        messagebox.showwarning(
            "Warning",
            "No CSV found ⚠\nTip: First click ‘Save Portfolio’ to create a CSV/Excel.",
        )
    set_status("Ready ✅")


def do_exit():
    """Clean exit from GUI."""
    root.quit()  # stops mainloop
    root.destroy()  # closes window


# ==============================
# Tkinter UI Setup
# ==============================
root = tk.Tk()
root.title("📊 Stock Portfolio Tracker")
root.geometry("1180x760")

# Nice ttk style
style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass

# --- Palette / Theme tweaks ---
style.configure("TButton", padding=6, font=("Segoe UI", 10))
style.configure("Treeview", font=("Segoe UI", 10), rowheight=26)
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
style.map("TButton", relief=[("pressed", "sunken"), ("active", "raised")])

# Top Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

ttk.Button(button_frame, text="🔄 View Portfolio (F5)", command=view_portfolio).grid(
    row=0, column=0, padx=5
)
ttk.Button(button_frame, text="➕ Add Stock (Ins)", command=ask_add).grid(
    row=0, column=1, padx=5
)
ttk.Button(button_frame, text="✏️ Edit Stock (Ctrl+E)", command=ask_edit).grid(
    row=0, column=2, padx=5
)
ttk.Button(button_frame, text="🗑️ Delete (Del)", command=ask_delete).grid(
    row=0, column=3, padx=5
)
ttk.Button(button_frame, text="💾 Save (Ctrl+S)", command=save_portfolio_gui).grid(
    row=0, column=4, padx=5
)
ttk.Button(button_frame, text="📂 Load (Ctrl+O)", command=load_portfolio_gui).grid(
    row=0, column=5, padx=5
)
ttk.Button(button_frame, text="📊 Charts", command=generate_charts_gui).grid(
    row=0, column=6, padx=5
)
ttk.Button(button_frame, text="🏆 Gainers/Losers", command=show_gainers_losers).grid(
    row=0, column=7, padx=5
)
ttk.Button(
    button_frame, text="🥧 Allocation Insights", command=show_allocation_insights
).grid(row=0, column=8, padx=5)
ttk.Button(button_frame, text="📈 Risk", command=show_risk_metrics).grid(
    row=0, column=9, padx=5
)
ttk.Button(button_frame, text="🧾 Transactions", command=view_transactions).grid(
    row=0, column=10, padx=5
)
ttk.Button(button_frame, text="🚪 Exit (Ctrl+Q)", command=do_exit).grid(
    row=0, column=11, padx=5
)

# Table
tree_frame = tk.Frame(root)
tree_frame.pack(pady=10, fill="both", expand=True)

columns = (
    "Ticker",
    "Quantity",
    "Buy Price",
    "Current Price",
    "Investment ($)",
    "Current Value ($)",
    "Profit/Loss ($)",
    "P/L %",
    "Weight %",
)
tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")

for col in columns:
    tree.heading(col, text=col)
    width = 150 if "($)" in col or col in ["Current Price", "Buy Price"] else 120
    tree.column(col, width=width, anchor="center")

# Zebra + P/L color tags
tree.tag_configure("even", background="#f9f9fc")
tree.tag_configure("odd", background="#ffffff")
tree.tag_configure("profit", foreground="#137333")  # Google green-ish
tree.tag_configure("loss", foreground="#b00020")  # Material red-ish

vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
tree.configure(yscroll=vsb.set)
tree.pack(side="left", fill="both", expand=True)
vsb.pack(side="right", fill="y")

# Status bar
status_var = tk.StringVar(value="Ready ✅")
status_bar = ttk.Label(root, textvariable=status_var, anchor="w")
status_bar.pack(fill="x", padx=8, pady=(0, 8))

# Keyboard shortcuts (speed!)
root.bind("<F5>", lambda e: view_portfolio())
root.bind("<Control-s>", lambda e: save_portfolio_gui())
root.bind("<Control-S>", lambda e: save_portfolio_gui())
root.bind("<Control-o>", lambda e: load_portfolio_gui())
root.bind("<Control-O>", lambda e: load_portfolio_gui())
root.bind("<Control-q>", lambda e: do_exit())
root.bind("<Control-Q>", lambda e: do_exit())
root.bind("<Insert>", lambda e: ask_add())
root.bind("<Delete>", lambda e: ask_delete())
root.bind("<Control-e>", lambda e: ask_edit())
root.bind("<Control-E>", lambda e: ask_edit())

# Clean window close (same as Exit)
root.protocol("WM_DELETE_WINDOW", do_exit)

# Initial load (with loader shown instantly)
view_portfolio()

# === RUN ===
# From terminal: Ctrl+C to stop if needed.
root.mainloop()
