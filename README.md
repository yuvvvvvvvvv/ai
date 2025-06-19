# Trading Bot Training System

## Project Overview
This project aims to train a trading bot for gold (XAUUSD) using advanced machine learning models. The system consists of several components:
- **Data Loading**: Efficiently loads and preprocesses historical trading data.
- **Model Training**: Implements LSTM, Transformer, and PPO models for predictive analysis and decision-making.
- **Risk Management**: Incorporates strategies to minimize trading risks.
- **Backtesting**: Evaluates model performance using historical data.

## File Structure
- **`train.py`**: Contains the training pipeline for executing model training and evaluation.
- **`ai_model.py`**: Defines model architectures and training methods for LSTM, Transformer, and PPO models.

## CSV File Format
The system expects CSV files with the following columns:
- `date`: Date in `YYYY-MM-DD` format.
- `time`: Time in `HH:MM:SS` format.
- `open`, `high`, `low`, `close`, `volume`: Standard trading data.
- `feature_0` to `feature_49`: Additional engineered features.

## Dependencies
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `tensorflow`
- `torch`
- `transformers`
- `tqdm`

## Usage Instructions
To run the training pipeline, execute:
```bash
python train.py
```
Expected outputs include trained model files (`trained_models.pt`) and log files (`bot_training.log`, `trading_bot.log`).

## Feature Engineering
`ai_model.py` adds technical indicators and cyclical features to enhance model predictions.

## Model Details
- **LSTM**: Used for time series prediction, capturing temporal dependencies.
- **Transformer**: Utilizes attention mechanisms for sequence prediction.
- **PPO**: Reinforcement learning model for decision-making in trading.

## Logging
Training and operational logs are stored in `bot_training.log` and `trading_bot.log` for monitoring and debugging purposes.