import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

class DataLoader:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        # Check for data_dir in config structure, with fallback to default 'data' directory
        self.data_dir = self.config.get('data', {}).get('data_dir', 'data')
        
        # Timeframe mappings for CSV files
        self.timeframe_files = {
            'M1': 'XAUUSD_M1.csv',  # Base timeframe
            'M5': 'XAUUSD_M5.csv',
            'M15': 'XAUUSD_M15.csv',
            'H1': 'XAUUSD_H1.csv',
            'H4': 'XAUUSD_H4.csv'
        }
        
        # Timeframe conversion factors (in minutes)
        self.timeframe_minutes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'H1': 60,
            'H4': 240
        }
    
    def load_csv_data(self, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """rical data from CSV file.
        
        Args:
            timeframe (str): Timeframe (e.g., 'M1', 'M5', 'M15', 'H1', 'H4')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Historical price data with OHLCV columns
        """
        try:
            print(f"Loading {timeframe} data from {self.data_dir}")
            print(f"Looking for file: {self.timeframe_files.get(timeframe)}")
            
            if timeframe not in self.timeframe_files:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
                
            file_path = Path(self.data_dir) / self.timeframe_files[timeframe]
            print(f"Full file path: {file_path}")
            print(f"File exists: {file_path.exists()}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
       r] = pd.to_dateti  qf =lpt Exceptimata  oeframe.
          Args:
            df (pd.DataFrame): Inputg'M5')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        try:
            if source_tf not in self.timeframe_minutes or target_tf not in self.timeframe_minutes:
                raise ValueError(f"Invalid timeframe: {source_tf} or {target_tf}")
            
            # Calculate resampling frequency
            freq = f"{selftet t all(    Resample OHLCm=or pleog( pen': 'f      'hhum'   }).droa
      lnfe
V    E ere
 "iSV.dsme)   fr('ep  n      # lna(method='ffill').f olumns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numasfloat) # ca      df = df[df['high'] >= df['low']]
            df = df[df['high'] >= df['open']]
            df = df[df['high'] >= df['close']]
            df = df[df['low'] <= df['            df = df[df['volume'] >= 0]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return pd.DataFrame()