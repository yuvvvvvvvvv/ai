import pandas as pd
import numpy as np
import logging
import asyncio
import psutil
from datetime import datetime, timedelta
import random
from ai_model import AIModel
from strategy_engine import StrategyEngine
from risk_manager import RiskManager

def generate_sample_data(num_days=5):
    """Generate synthetic market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=num_days*24*60, freq='1min')
    base_price = 1900 + random.uniform(-100, 100)
    
    data = {
        'time': dates,
        'open': [base_price + random.uniform(-5, 5) for _ in range(len(dates))],
        'high': [base_price + random.uniform(0, 10) for _ in range(len(dates))],
        'low': [base_price + random.uniform(-10, 0) for _ in range(len(dates))],
        'close': [base_price + random.uniform(-5, 5) for _ in range(len(dates))],
        'volume': [random.randint(100, 1000) for _ in range(len(dates))],
        'rsi': [random.uniform(30, 70) for _ in range(len(dates))],
        'macd': [random.uniform(-5, 5) for _ in range(len(dates))],
        'adx': [random.uniform(20, 50) for _ in range(len(dates))]
    }
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    return df

def generate_sample_news():
    """Generate sample news articles for testing"""
    headlines = [
        "Gold prices rise on inflation fears",
        "Fed signals potential rate hike",
        "Geopolitical tensions boost safe-haven demand",
        "Strong dollar pressures gold prices",
        "Central banks increase gold reserves"
    ]
    return [{'title': random.choice(headlines), 'description': 'Sample news description'} for _ in range(3)]

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_fetcher():
    """Test data fetcher functionality (mocked, no MT5)"""
    logger.info("Testing Data Fetcher (mocked, no MT5)...")
    class MockDataFetcher:
        def get_historical_data(self):
            return generate_sample_data(5)
        def get_tick_data(self):
            return generate_sample_data(1).tail(1)
        def get_order_book(self):
            return {
                'bids': [{'volume': 100, 'price': 1900}],
                'asks': [{'volume': 120, 'price': 1901}]
            }
    data_fetcher = MockDataFetcher()
    try:
        historical_data = data_fetcher.get_historical_data()
        assert historical_data is not None, "Historical data fetch failed"
        logger.info("Historical data fetch successful (mocked)")
        tick_data = data_fetcher.get_tick_data()
        assert tick_data is not None, "Tick data fetch failed"
        logger.info("Tick data fetch successful (mocked)")
        order_book = data_fetcher.get_order_book()
        assert order_book is not None, "Order book fetch failed"
        logger.info("Order book fetch successful (mocked)")
        return data_fetcher
    except Exception as e:
        logger.error(f"Data Fetcher test failed: {e}")
        return None

def test_ai_model(df):
    """Test AI model functionality"""
    logger.info("Testing AI Model...")
    ai_model = AIModel()
    
    try:
        # Test feature engineering
        df_features = ai_model.engineer_features(df)
        assert df_features is not None, "Feature engineering failed"
        logger.info("Feature engineering successful")
        
        # Test model training
        training_success = ai_model.train_models(df)
        assert training_success, "Model training failed"
        logger.info("Model training successful")
        
        # Test price prediction
        prediction = ai_model.predict_price(df.tail(LSTM_SEQUENCE_LENGTH + 1))
        assert prediction is not None, "Price prediction failed"
        logger.info(f"Price Prediction: {prediction}")
        
        # Test sentiment analysis
        news_texts = generate_sample_news()
        sentiment = ai_model.analyze_sentiment(news_texts)
        assert sentiment is not None, "Sentiment analysis failed"
        logger.info(f"Sentiment Score: {sentiment}")
        
        return ai_model
    except Exception as e:
        logger.error(f"AI Model test failed: {e}")
        return None

def test_strategy_engine(df):
    """Test strategy engine functionality"""
    logger.info("Testing Strategy Engine...")
    strategy = StrategyEngine()
    
    try:
        # Test technical analysis
        df_indicators = strategy.calculate_indicators(df)
        assert df_indicators is not None, "Technical analysis failed"
        logger.info("Technical indicators calculated successfully")
        
        # Test trend signals
        trend_signals = strategy.detect_trend_signals(df_indicators)
        assert trend_signals is not None, "Trend signal detection failed"
        logger.info(f"Trend Signals: {trend_signals}")
        
        # Test mean reversion signals
        mr_signals = strategy.detect_mean_reversion_signals(df_indicators)
        assert mr_signals is not None, "Mean reversion signal detection failed"
        logger.info(f"Mean Reversion Signals: {mr_signals}")
        
        # Test breakout signals
        breakout_signals = strategy.detect_breakout_signals(df_indicators)
        assert breakout_signals is not None, "Breakout signal detection failed"
        logger.info(f"Breakout Signals: {breakout_signals}")
        
        # Test integrated signal generation
        order_book = {
            'bids': [{'volume': 100}, {'volume': 150}],
            'asks': [{'volume': 120}, {'volume': 130}]
        }
        signals = strategy.generate_signals(df_indicators, order_book)
        assert signals is not None, "Signal generation failed"
        logger.info(f"Generated Signal: {signals}")
        
        return strategy
    except Exception as e:
        logger.error(f"Strategy Engine test failed: {e}")
        return None

def test_risk_manager():
    """Test risk manager functionality"""
    logger.info("Testing Risk Manager...")
    risk_manager = RiskManager()
    
    try:
        # Create mock account info
        class MockAccountInfo:
            def __init__(self, equity):
                self.equity = equity
                self.balance = equity
                self.margin = 0
                self.margin_free = equity
        
        # Test position sizing
        account_info = MockAccountInfo(100000)
        entry_price = 1900
        stop_loss = 1890
        
        position_size = risk_manager.calculate_position_size(
            account_info, entry_price, stop_loss
        )
        assert position_size > 0, "Position sizing calculation failed"
        logger.info(f"Calculated Position Size: {position_size}")
        
        # Test portfolio metrics with mock portfolio
        mock_portfolio = {
            'equity': account_info.equity,
            'positions': [
                {'symbol': 'GOLD', 'size': position_size, 'entry_price': entry_price}
            ]
        }
        portfolio_metrics = risk_manager.calculate_portfolio_metrics(mock_portfolio)
        assert portfolio_metrics is not None, "Portfolio metrics calculation failed"
        logger.info(f"Portfolio Metrics: {portfolio_metrics}")
        
        # Test risk limits
        risk_check = risk_manager.check_risk_limits(position_size, entry_price)
        assert risk_check is not None, "Risk limit check failed"
        logger.info(f"Risk Check Result: {risk_check}")
        
        return risk_manager
    except Exception as e:
        logger.error(f"Risk Manager test failed: {e}")
        return None

async def test_trading_bot():
    """Test trading bot functionality and error recovery"""
    logger.info("Testing Trading Bot...")
    trading_bot = TradingBot()
    
    try:
        # Test component initialization
        assert all(trading_bot.component_status.values()), "Component initialization failed"
        logger.info("All components initialized successfully")
        
        # Test system health monitoring
        metrics = trading_bot.performance_metrics['system_metrics']
        assert len(metrics) > 0, "System health monitoring failed"
        logger.info(f"System Metrics: {metrics[-1]}")
        
        # Test degraded mode
        trading_bot.enter_degraded_mode('ai_model')
        assert trading_bot.degraded_mode, "Degraded mode transition failed"
        logger.info("Degraded mode test successful")
        
        # Test data processing
        sample_data = {
            'tick_data': generate_sample_data(1),
            'order_book': {'bids': [{'volume': 100}], 'asks': [{'volume': 100}]},
            'news': generate_sample_news(),
            'economic_calendar': [],
            'alternative_data': {}
        }
        await trading_bot.process_data(sample_data)
        logger.info("Data processing test successful")
        
        return trading_bot
    except Exception as e:
        logger.error(f"Trading Bot test failed: {e}")
        return None

async def main():
    """Main testing function"""
    try:
        # Generate sample data
        logger.info("Generating sample data...")
        df = generate_sample_data()
        
        # Test individual components
        data_fetcher = test_data_fetcher()
        ai_model = test_ai_model(df)
        strategy = test_strategy_engine(df)
        risk_manager = test_risk_manager()
        
        # Verify component tests
        components = [data_fetcher, ai_model, strategy, risk_manager]
        assert all(component is not None for component in components), "One or more component tests failed"
        
        # Test trading bot integration
        trading_bot = await test_trading_bot()
        assert trading_bot is not None, "Trading bot integration test failed"
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())