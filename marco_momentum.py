import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Momentum Trading Strategy", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e6e6e6;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Momentum-Based Trading Strategy")

# Explanation text
st.markdown(
    """
    This app implements a momentum-based trading strategy. The strategy invests in the top `n` assets with the highest momentum
    at the end of each month. The momentum is calculated based on a 12-month period using the `13612W` formula:

    ### Formula
    The formula used to calculate the momentum is defined as follows:
    $$
    \\text{Momentum} = 12 \\times \\left(\\frac{p_0}{p_1} - 1\\right) + 4 \\times \\left(\\frac{p_0}{p_3} - 1\\right) + 2 \\times \\left(\\frac{p_0}{p_6} - 1\\right) + \\left(\\frac{p_0}{p_{12}} - 1\\right)
    $$

    - $p_0$: The most recent closing price.
    - $p_1, p_3, p_6, p_{12}$: Closing prices 1 month, 3 months, 6 months, and 12 months ago respectively.

    The strategy is implemented on a selection of assets, and the portfolio is rebalanced monthly. Below, you can see how the strategy performs over time, as well as the current composition of the portfolio based on the latest momentum calculations.
    """
)

# Sidebar for user input
st.sidebar.header("Input Parameters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
end_date = pd.to_datetime("today")  # Set end date to today
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000)
top_n_assets = st.sidebar.number_input("Number of Top Momentum Assets", value=3, min_value=1, max_value=10)

# Define the stocks and download data
symbols = ["USEU.MI", "EUUS.MI", "LJPY.L", "SJPY.L", "GBUS.L", "USGB.L"]
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Fill missing values
data = data.fillna(method='ffill')

# Convert data index to naive datetime to avoid timezone issues
data.index = data.index.tz_localize(None)

# Calculate momentum for each asset
def calculate_momentum(prices):
    if len(prices) < 252:
        return np.nan  # Not enough data for 252-day momentum
    p0 = prices.iloc[-1]
    p1 = prices.iloc[-21]
    p3 = prices.iloc[-63]
    p6 = prices.iloc[-126]
    p12 = prices.iloc[-252]
    momentum = (12 * (p0 / p1 - 1)) + (4 * (p0 / p3 - 1)) + (2 * (p0 / p6 - 1)) + (p0 / p12 - 1)
    return momentum

# Apply momentum calculation daily
momentum = data.rolling(window=252).apply(calculate_momentum, raw=False)

# Display initial momentum values for all assets
st.subheader("Initial Momentum Values for All Assets")
st.markdown("The initial momentum values are calculated for each asset in the portfolio. These values give an indication of the relative strength of each asset over the past year.")
st.write(momentum.dropna().iloc[0].sort_values(ascending=False))

# Initialize portfolio variables
portfolio_value = initial_capital
portfolio_values = [initial_capital]
portfolio_dates = []
current_portfolio = None

# Define portfolio rebalancing dates (last trading day of each month)
rebalancing_dates = data.resample('M').last().index

# Ensure the backtest starts after we have enough data to calculate momentum
momentum_start_date = momentum.dropna().index[0] if not momentum.dropna().empty else None
if momentum_start_date:
    rebalancing_dates = rebalancing_dates[rebalancing_dates > momentum_start_date]

# Backtest button
if st.button("Run Backtest"):

    # Run backtest
    for date in rebalancing_dates:
        if date not in momentum.index or momentum.loc[date].isnull().all():
            continue  # Skip months where momentum can't be calculated

        top_assets = momentum.loc[date].dropna().nlargest(top_n_assets).index

        if len(top_assets) < top_n_assets:
            continue  # If less than top_n_assets have valid momentum, skip the period

        # Calculate the monthly return of the portfolio
        month_data = data.loc[date-pd.offsets.MonthBegin(1):date][top_assets]
        monthly_returns = month_data.pct_change().dropna().mean(axis=1)

        if monthly_returns.empty:
            continue  # Skip if no valid data is available

        cumulative_return = (1 + monthly_returns).prod() - 1
        portfolio_value *= (1 + cumulative_return)
        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

        # Update current portfolio
        current_portfolio = top_assets

    # Plot the cumulative returns using Plotly
    st.subheader("Cumulative Returns")
    st.markdown("The cumulative returns chart displays the growth of the initial investment over time based on the momentum strategy. The portfolio is rebalanced at the end of each month to include the top assets by momentum.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_dates, y=portfolio_values[1:], mode='lines', name='Cumulative Returns'))
    fig.update_layout(title='Cumulative Returns of Momentum Strategy', xaxis_title='Date', yaxis_title='Portfolio Value ($)', height=600, width=900)
    st.plotly_chart(fig)

    # Display the last values of the current momentum
    st.subheader("Last Values of Current Momentum")
    st.markdown("The last values of the current momentum are shown below. These values indicate which assets are currently exhibiting the highest momentum, making them the top candidates for inclusion in the portfolio.")
    last_momentum = momentum.iloc[-1].dropna().nlargest(top_n_assets)
    st.write(last_momentum)

    # Calculate the overall MTD % of the PnL of the portfolio
    current_month_start = pd.to_datetime(f"{end_date.year}-{end_date.month:02d}-01").tz_localize(None)
    portfolio_mtd_return = (portfolio_values[-1] / portfolio_values[-2] - 1) * 100 if len(portfolio_values) > 1 else 0
    st.write(f"Overall Portfolio MTD Return: {portfolio_mtd_return:.2f}%")

    # Print the current portfolio and MTD % for each asset using Plotly table
    st.subheader("Current Portfolio Composition and MTD %")
    st.markdown("The current portfolio composition lists the top assets based on the latest momentum calculation. The Month-To-Date (MTD) % shows the performance of each asset since the beginning of the current month.")
    if current_portfolio is not None:
        mtd_asset_returns = {}
        for asset in current_portfolio:
            asset_data = data.loc[current_month_start:end_date, asset]
            if not asset_data.empty:
                asset_mtd_return = (asset_data.pct_change().dropna() + 1).prod() - 1
                mtd_asset_returns[asset] = asset_mtd_return * 100

        # Plot MTD % returns for each asset
        mtd_fig = go.Figure(data=[go.Bar(x=list(mtd_asset_returns.keys()), y=list(mtd_asset_returns.values()))])
        mtd_fig.update_layout(title='MTD % Returns of Current Portfolio Assets', xaxis_title='Asset', yaxis_title='MTD % Return', height=400, width=700)
        st.plotly_chart(mtd_fig)

        st.write(pd.Series(mtd_asset_returns).to_frame(name="MTD %"))
    else:
        st.write("No valid portfolio composition found.")

    # Predict next month's top assets
    st.subheader("Predicted Top Assets for Next Month")
    st.markdown("Based on the current momentum values, the predicted top assets for the next month are displayed below. These are the assets that will most likely be included in the portfolio for the upcoming month.")
    predicted_top_assets = momentum.iloc[-1].dropna().nlargest(top_n_assets)
    st.write(predicted_top_assets)
