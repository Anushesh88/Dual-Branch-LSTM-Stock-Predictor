# Dual Branch LSTM Stock Price Predictor

## 📌 Project Overview
This project implements a **Dual-Branch LSTM (Long Short-Term Memory)** neural network to predict stock prices. Unlike traditional single-input models, this architecture splits features into two distinct branches—**Macro (Trend)** and **Micro (Volatility/Momentum)**—to capture different market dynamics before fusing them for a final prediction.

The model is built using **PyTorch** and trained on historical stock data fetched via `yfinance`.

## 🚀 Key Features
* **Dual-Branch Architecture:**
    * **Trend Branch:** Processes long-term indicators (SMA, MACD).
    * **Momentum Branch:** Processes short-term volatility indicators (RSI, ATR, Bollinger Bands).
* **Automated Data Fetching:** Downloads real-time data using the Yahoo Finance API.
* **Advanced Feature Engineering:** Calculates technical indicators like RSI, MACD, and Bollinger Bands from scratch.
* **Robust Training Pipeline:** Includes learning rate scheduling (`ReduceLROnPlateau`) and Early Stopping to prevent overfitting.

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Deep Learning:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Data Source:** yfinance
* **Preprocessing:** Scikit-Learn (StandardScaler)
* **Visualization:** Matplotlib (implicit in notebook usage)

## 📊 Model Architecture
The model consists of two parallel LSTM layers:
1.  **Branch 1 (Trend):** Takes `SMA_50`, `SMA_200`, `MACD` signal lines.
2.  **Branch 2 (Momentum):** Takes `RSI`, `ATR`, and `Bollinger Bands`.
3.  **Fusion Layer:** Concatenates the outputs of both LSTMs.
4.  **Fully Connected Head:** A sequential neural network that maps the fused features to a single predicted stock price.

## 📂 Dataset
The project uses historical daily data for **Canara Bank (CANBK.NS)**, but it can be easily adapted for any stock ticker supported by Yahoo Finance.

## ⚙️ How to Run
1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy yfinance scikit-learn torchinfo
    ```
2.  **Open the Notebook:**
    Launch Jupyter Notebook or Google Colab and open `Dual_Branch_LSTM_Stock_Price_Predictor.ipynb`.
3.  **Execute Cells:**
    Run the cells in order to fetch data, process features, build the model, and train it.

## 📈 Results
The model is evaluated using **Mean Squared Error (MSE)** loss. It outputs the predicted closing price for the next trading day based on a lookback window of historical data.

## 📜 License
This project is open-source and available for educational purposes.
