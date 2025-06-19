import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from ai_model import AIModel
from data_preparation import DataPreparation
from backtesting import BacktestEngine
from risk_manager import RiskManager
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, sequence_length):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='bot_training.log'
)
logger = logging.getLogger(__name__)

class GoldBotTrainer:
    def __init__(self, config):
        self.config = config
        self.data_prep = DataPreparation()
        self.ai_model = AIModel()
        self.backtester = None
        self.risk_manager = RiskManager()
        
    def load_and_prepare_data(self):
        """Load data from CSV files and prepare for training"""
        try:
            # Load data from CSV files
            data = {}
            timeframes = ['5M', '15M', '1H', '4H']
            required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume'] + [f'feature_{i}' for i in range(50)]
            critical_columns = ['open', 'high', 'low', 'close', 'volume']

            for tf in timeframes:
                file_path = f'data/merged_{tf}_2024_to_2015.csv'
                # Read CSV with specific column names
                df = pd.read_csv(file_path, names=required_columns, header=None)

                # Validation checks
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing columns in {file_path}: {missing_columns}")
                    raise ValueError(f"Missing columns in {file_path}: {missing_columns}")

                if df.empty:
                    logger.error(f"DataFrame is empty for {file_path}")
                    raise ValueError(f"DataFrame is empty for {file_path}")

                if df[critical_columns].isnull().any().any():
                    logger.error(f"NaN values found in critical columns for {file_path}")
                    raise ValueError(f"NaN values found in critical columns for {file_path}")

                # Combine date and time to create timestamp
                try:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
                except Exception as e:
                    logger.error(f"Failed to parse timestamp for {file_path}: {str(e)}")
                    raise ValueError(f"Failed to parse timestamp for {file_path}: {str(e)}")

                df.set_index('timestamp', inplace=True)

                # Drop original date and time columns
                df.drop(['date', 'time'], axis=1, inplace=True)

                data[tf] = df

            # Prepare data for different models
            prepared_data = {
                'train': {
                    'M5': data['5M'],
                    'H1': data['1H'],
                    'tick': data['5M']  # Using 5M data for tick simulation
                },
                'validation': data['15M'],
                'test': data['4H']
            }

            logger.info("Data loaded and prepared successfully")
            return prepared_data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def train_models(self, data):
        """Train all AI components with validation"""
        training_plan = [
            {
                'name': 'LSTM',
                'function': self.ai_model.train_lstm_model,
                'data': data['train']['M5'],
                'params': {
                    'epochs': 100,
                    'batch_size': 64,
                    'validation_split': 0.2
                }
            },
            {
                'name': 'Transformer',
                'function': self.ai_model.train_transformer_model,
                'data': data['train']['H1'],
                'params': {
                    'epochs': 50,
                    'batch_size': 32
                }
            },
            {
                'name': 'PPO',
                'function': self.ai_model.train_ppo_agent,
                'data': data['train']['tick'],
                'params': {
                    'episodes': 1000,
                    'gamma': 0.99,
                    'epsilon': 0.2
                }
            }
        ]

        def train_job(job):
            try:
                logger.info(f"Training {job['name']} model...")
                if job['name'] == 'Transformer':
                    # Fit StandardScaler to the training data
                    features = job['data'][LSTM_FEATURES].values
                    self.ai_model.feature_scaler.fit(features)
                job['function'](job['data'], **job['params'])
                logger.info(f"{job['name']} training completed")
            except Exception as e:
                logger.error(f"{job['name']} training failed: {str(e)}")
                return job['name'], str(e)
            return job['name'], None

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(train_job, job) for job in training_plan]
            failed_jobs = []
            for future in tqdm(futures, desc="Training models"):
                name, error = future.result()
                if error:
                    failed_jobs.append((name, error))

        if failed_jobs:
            error_messages = ", ".join([f"{name} failed: {error}" for name, error in failed_jobs])
            raise Exception(f"Training failed for models: {error_messages}")

    def optimize_risk_parameters(self, data):
        """Calibrate risk management thresholds"""
        params_to_optimize = {
            'volatility_threshold': (1.0, 3.0),
            'max_drawdown': (0.05, 0.20),
            'position_size_multiplier': (0.5, 2.0)
        }
        
        best_params = {}
        for param, (min_val, max_val) in tqdm(params_to_optimize.items(), desc="Optimizing risk"):
            best_score = -np.inf
            for value in np.linspace(min_val, max_val, 10):
                setattr(self.risk_manager, param, value)
                backtest_results = self.backtester.run(data['validation'])
                score = backtest_results['sharpe_ratio']
                
                if score > best_score:
                    best_score = score
                    best_params[param] = value
                    
        logger.info(f"Optimized risk params: {best_params}")
        return best_params

    def run_walk_forward_validation(self, full_data):
        """Walk-forward validation across multiple regimes"""
        regime_periods = {
            'low_volatility': ('2020-01-01', '2020-03-01'),
            'high_volatility': ('2020-03-01', '2020-06-01'),
            'bull_market': ('2020-06-01', '2020-12-01')
        }
        
        results = {}
        for regime, (start, end) in tqdm(regime_periods.items(), desc="Walk-forward validation"):
            regime_data = full_data.loc[start:end]
            self.backtester = BacktestEngine(regime_data)
            results[regime] = self.backtester.run_backtest()
            
        return results

    def save_models(self):
        """Save all trained components"""
        components = {
            'lstm': self.ai_model.lstm_model,
            'transformer': self.ai_model.transformer_model,
            'ppo': self.ai_model.ppo_agent,
            'risk_params': self.risk_manager.get_parameters()
        }
        
        torch.save(components, 'trained_models.pt')
        logger.info("All models saved successfully")

    def full_training_pipeline(self):
        """End-to-end training workflow"""
        try:
            # Step 1: Data Loading
            raw_data = self.load_and_prepare_data()
            
            # Step 2: Train AI Models
            self.train_models(raw_data)
            
            # Step 3: Risk Optimization
            optimized_params = self.optimize_risk_parameters(raw_data)
            self.risk_manager.update_parameters(optimized_params)
            
            # Step 4: Validation
            validation_results = self.run_walk_forward_validation(raw_data['test'])
            
            # Step 5: Final Save
            self.save_models()
            
            logger.info("Training pipeline completed successfully")
            return {
                'validation_results': validation_results,
                'risk_parameters': optimized_params
            }
            
        except Exception as e:
            logger.critical(f"Training pipeline failed: {str(e)}")
            raise

# Usage Example
if __name__ == "__main__":
    config = {
        'data': {
            'start_date': '2015-01-01',
            'end_date': '2023-12-31',
            'timeframes': ['5M', '15M', '1H', '4H']
        }
    }
    
    trainer = GoldBotTrainer(config)
    training_report = trainer.full_training_pipeline()
    print("Training completed with metrics:", training_report)

# Update train_models method to use DataGenerator
self.ai_model.train_lstm_model(data['train']['M5'], epochs=100, batch_size=64, validation_split=0.2)