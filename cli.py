import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import time
import sys

# ----- Terminal Coloring -----
HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
BOLD = "\033[1m"
ENDC = "\033[0m"

# ----- Portfolio Data -----
portfolio = {
    "AAPL": {"quantity": 10, "buy_price": 150},
    "MSFT": {"quantity": 5, "buy_price": 300},
    "GOOGL": {"quantity": 2, "buy_price": 2500},
}


# ----- Loading Animation -----
def loading_animation(text="Processing", duration=2):
    sys.stdout.write(f"{OKCYAN}{text}")
    sys.stdout.flush()
    for _ in range(duration * 3):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.33)
    print(f"{ENDC}")


# ----- Portfolio Calculation -----
def calculate_portfolio():
    loading_animation("Fetching live stock data", 2)
    portfolio_data = []
    for ticker, details in portfolio.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty or "Close" not in hist.columns:
            print(f"{WARNING}⚠ No valid data for {ticker}, skipping...{ENDC}")
            continue
        current_price = hist["Close"].iloc[-1]
        quantity = details["quantity"]
        buy_price = details["buy_price"]
        investment = buy_price * quantity
        current_value = current_price * quantity
        profit_loss = current_value - investment
        portfolio_data.append(
            {
                "Ticker": ticker,
                "Quantity": quantity,
                "Buy Price": buy_price,
                "Current Price": round(current_price, 2),
                "Investment ($)": round(investment, 2),
                "Current Value ($)": round(current_value, 2),
                "Profit/Loss ($)": round(profit_loss, 2),
            }
        )
    return pd.DataFrame(portfolio_data)


# ----- Display Portfolio Table with Colors -----
def display_table(df):
    if df.empty:
        print(f"{FAIL}⚠ Portfolio is empty!{ENDC}")
        return

    headers = [
        f"{BOLD}{OKBLUE}Ticker{ENDC}",
        f"{BOLD}{OKBLUE}Quantity{ENDC}",
        f"{BOLD}{OKBLUE}Buy Price{ENDC}",
        f"{BOLD}{OKBLUE}Current Price{ENDC}",
        f"{BOLD}{OKBLUE}Investment ($){ENDC}",
        f"{BOLD}{OKBLUE}Current Value ($){ENDC}",
        f"{BOLD}{OKBLUE}Profit/Loss ($){ENDC}",
    ]

    # Color data: green for profit, red for loss
    colored_data = []
    for _, row in df.iterrows():
        pl_color = OKGREEN if row["Profit/Loss ($)"] >= 0 else FAIL
        colored_data.append(
            [
                row["Ticker"],
                row["Quantity"],
                row["Buy Price"],
                row["Current Price"],
                row["Investment ($)"],
                row["Current Value ($)"],
                f"{pl_color}{row['Profit/Loss ($)']}{ENDC}",
            ]
        )

    print(f"\n{BOLD}{OKCYAN}📊 Portfolio Report:{ENDC}\n")
    print(tabulate(colored_data, headers=headers, tablefmt="fancy_grid"))

    # Portfolio Summary
    total_investment = df["Investment ($)"].sum()
    total_current_value = df["Current Value ($)"].sum()
    total_profit_loss = df["Profit/Loss ($)"].sum()
    summary_data = [
        ["Total Investment ($)", total_investment],
        ["Total Current Value ($)", total_current_value],
        [
            "Total Profit/Loss ($)",
            f"{OKGREEN if total_profit_loss>=0 else FAIL}{total_profit_loss}{ENDC}",
        ],
    ]
    print(f"\n{BOLD}{OKGREEN}📊 Portfolio Summary:{ENDC}\n")
    print(
        tabulate(
            summary_data,
            headers=[f"{BOLD}Metric{ENDC}", f"{BOLD}Value{ENDC}"],
            tablefmt="fancy_grid",
        )
    )


# ----- Generate Charts -----
def generate_charts(df):
    if df.empty:
        print(f"{FAIL}⚠ Portfolio empty, cannot generate charts!{ENDC}")
        return
    choice = (
        input(
            f"{OKBLUE}Do you want to save charts as PNG or show in terminal? (save/show): {ENDC}"
        )
        .strip()
        .lower()
    )

    # Bar chart
    plt.figure(figsize=(8, 5))
    sns.set_style("darkgrid")
    sns.barplot(
        x="Ticker",
        y="Profit/Loss ($)",
        data=df,
        hue="Ticker",
        dodge=False,
        legend=False,
        palette="coolwarm",
    )
    plt.title("Profit/Loss per Stock", fontsize=14, fontweight="bold")
    plt.xlabel("Stock Ticker", fontsize=12)
    plt.ylabel("Profit/Loss ($)", fontsize=12)
    plt.axhline(0, color="black", linewidth=1)
    if choice == "save":
        bar_file = "profit_loss_chart.png"
        plt.savefig(bar_file, bbox_inches="tight")
        print(f"{OKGREEN}✅ Bar chart saved as {bar_file}{ENDC}")
        plt.close()
    else:
        plt.show()

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        df["Current Value ($)"],
        labels=df["Ticker"],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Portfolio Distribution by Value", fontsize=14, fontweight="bold")
    if choice == "save":
        pie_file = "portfolio_distribution.png"
        plt.savefig(pie_file, bbox_inches="tight")
        print(f"{OKGREEN}✅ Pie chart saved as {pie_file}{ENDC}")
        plt.close()
    else:
        plt.show()


# ----- Save Portfolio -----
def save_portfolio(df):
    df.to_csv("portfolio_report.csv", index=False)
    df.to_excel("portfolio_report.xlsx", index=False)
    print(f"{OKGREEN}✅ Portfolio saved to CSV & Excel{ENDC}")


# ----- Load Portfolio -----
def load_portfolio():
    if os.path.exists("portfolio_report.csv"):
        df = pd.read_csv("portfolio_report.csv")
        print(f"{OKCYAN}📂 Loaded Portfolio from CSV!{ENDC}")
        return df
    else:
        print(f"{WARNING}⚠ No CSV file found! Calculating live portfolio...{ENDC}")
        return calculate_portfolio()


# ----- Main Menu -----
def main():
    df = calculate_portfolio()  # Initial calculation
    while True:
        print(f"\n{BOLD}{OKBLUE}--- Stock Portfolio Tracker Menu ---{ENDC}")
        print("1. View Portfolio Table")
        print("2. Generate / Update Charts")
        print("3. Save Portfolio to CSV/Excel")
        print("4. Load Portfolio from CSV")
        print("5. Exit")
        choice = input(f"{OKCYAN}Enter choice (1-5): {ENDC}").strip()

        if choice == "1":
            if df.empty:
                df = calculate_portfolio()
            display_table(df)
        elif choice == "2":
            if df.empty:
                df = calculate_portfolio()
            generate_charts(df)
        elif choice == "3":
            if df.empty:
                df = calculate_portfolio()
            save_portfolio(df)
        elif choice == "4":
            df = load_portfolio()
        elif choice == "5":
            print(f"{OKGREEN}Exiting... 👋{ENDC}")
            break
        else:
            print(f"{FAIL}⚠ Invalid choice!{ENDC}")


if __name__ == "__main__":
    main()
