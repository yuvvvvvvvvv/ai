import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Technical Analysis Constants
EMA_FAST = 12
EMA_SLOW = 26
ADX_PERIOD = 14
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Risk Management Constants
MAX_POSITION_SIZE = 5.0  # Maximum position size in lots
MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_EXPOSURE_PER_TRADE = 0.05  # 5% max exposure per trade
ATR_MULTIPLIER = 2.0  # Multiplier for ATR-based stop loss
SYMBOL = 'XAUUSD'  # Trading symbol

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path (str): Path to config YAML file.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Set default values if not present
            config.setdefault('TIMEFRAMES', ['M5', 'M15', 'H1', 'H4'])
            config.setdefault('DEVICE', 'cpu')
            config.setdefault('TASK', 'regression')
            config.setdefault('LEARNING_RATE', 0.001)
            config.setdefault('EPOCHS', 100)
            config.setdefault('BATCH_SIZE', 32)
            config.setdefault('LSTM_UNITS', 50)
            config.setdefault('TRANSFORMER_LAYERS', 3)
            config.setdefault('ATTENTION_HEADS', 8)
            
            # Ensure model paths exist
            config.setdefault('LSTM_MODEL_PATH', 'models/lstm_model')
            config.setdefault('TRANSFORMER_MODEL_PATH', 'models/transformer_model')
            config.setdefault('PPO_MODEL_PATH', 'models/ppo_model')
            config.setdefault('FINBERT_MODEL_PATH', 'models/finbert_model')
            
            # Create model directories if they don't exist
            Path('models').mkdir(exist_ok=True)
            
            return config
            
        except FileNotFoundError:
            logging.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading config: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = 'config.yaml') -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to save.
            config_path (str): Path to save config YAML file.
            
        Raises:
            yaml.YAMLError: If config cannot be serialized
            PermissionError: If file cannot be written
        """
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
                
            logging.info(f"Configuration saved to {config_path}")
            
        except yaml.YAMLError as e:
            logging.error(f"Error serializing config: {e}")
            raise
        except PermissionError:
            logging.error(f"Permission denied writing to {config_path}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error saving config: {e}")
            raise