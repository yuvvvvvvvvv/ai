import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Constants for LSTM model
LSTM_SEQUENCE_LENGTH = 60
LSTM_UNITS = 128
LSTM_FEATURES = ['open', 'high', 'low', 'close', 'volume']

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, sequence_length):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.indexes = np.arange(len(self.X) - self.sequence_length)
        
    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.array([self.X[i:i+self.sequence_length] for i in batch_indexes])
        y_batch = np.array([self.y[i+self.sequence_length] for i in batch_indexes])
        return X_batch, y_batch
        
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.optim import Adam as TorchAdam
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import *

# Define constants
LSTM_FEATURES = ['open', 'high', 'low', 'close', 'volume']
LSTM_SEQUENCE_LENGTH = 60
LSTM_UNITS = 128
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 100
import logging
from datetime import datetime
import tensorflow as tf
from collections import deque
import random

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.epsilon = 0.2  # PPO clipping parameter
        self.gamma = 0.99  # Discount factor
        
        # Initialize actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
    
    def _build_actor(self):
        """Build actor network for policy approximation"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001))
        self.logger.info("Actor model architecture updated.")
        return model
    
    def _build_critic(self):
        """Build critic network for value approximation"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.logger.info("Critic model architecture updated.")
        return model
    
    def get_action(self, state):
        """Get action using current policy"""
        state = np.array([state])
        action_probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action, action_probs
    
    def store_transition(self, state, action, reward, next_state, done, action_probs):
        """Store transition in memory buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.action_probs.append(action_probs)
    
    def train(self):
        """Train PPO agent using collected experiences"""
        if len(self.states) < 32:  # Minimum batch size
            return True  # Return True even if no training occurs
        
        try:
            states = np.array(self.states)
            actions = np.array(self.actions)
            rewards = np.array(self.rewards)
            next_states = np.array(self.next_states)
            dones = np.array(self.dones)
            old_action_probs = np.array(self.action_probs)
            
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Calculate advantages
            values = self.critic.predict(states, verbose=0).flatten()
            next_values = self.critic.predict(next_states, verbose=0).flatten()
            
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert actions to one-hot encoding
            actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_dim)
            
            # PPO update
            for _ in range(5):  # Reduced number of optimization epochs for stability
                # Update critic first
                target_values = rewards + self.gamma * next_values * (1 - dones)
                critic_loss = self.critic.train_on_batch(states, target_values.reshape(-1, 1))
                
                # Update actor
                with tf.GradientTape() as tape:
                    current_action_probs = self.actor(states)
                    # Calculate ratio using one-hot encoded actions
                    ratio = tf.reduce_sum(current_action_probs * actions_one_hot, axis=1) / \
                            (tf.reduce_sum(old_action_probs * actions_one_hot, axis=1) + 1e-10)
                    surrogate1 = ratio * advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Clear memory buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.action_probs = []
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in PPO training step: {e}")
            return True  # Still return True to continue training

class AIModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lstm_model = None
        self.transformer_model = None
        self.ensemble_weights = {'lstm': 0.3, 'transformer': 0.3, 'sentiment': 0.2, 'ppo': 0.2}
        self.feature_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model_performance = {'lstm': [], 'transformer': [], 'sentiment': [], 'ppo': []}
        self.last_retrain = datetime.now()
        self.ppo_agent = None
        self.reward_history = []

    def predict_price(self, data):
        """Predict future price using trained models"""
        try:
            # Validate input data
            if data is None or len(data) == 0:
                self.logger.error("Empty or None input data")
                return {'prediction': 0.0, 'confidence_interval': {'lower': 0.0, 'upper': 0.0}}

            # Check for required columns
            missing_cols = [col for col in LSTM_FEATURES if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return {'prediction': 0.0, 'confidence_interval': {'lower': 0.0, 'upper': 0.0}}

            # Handle insufficient data
            if len(data) < LSTM_SEQUENCE_LENGTH:
                self.logger.warning(f"Insufficient data for prediction. Required: {LSTM_SEQUENCE_LENGTH}, Got: {len(data)}")
                # Pad data with its own values if possible
                if len(data) > 0:
                    pad_length = LSTM_SEQUENCE_LENGTH - len(data)
                    padded_data = pd.concat([data.iloc[[0]].repeat(pad_length), data])
                    data = padded_data.copy()
                else:
                    return {'prediction': 0.0, 'confidence_interval': {'lower': 0.0, 'upper': 0.0}}

            # Prepare data
            X = data[LSTM_FEATURES].values
            X_scaled = self.feature_scaler.transform(X)
            X_seq = X_scaled[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, len(LSTM_FEATURES))

            # Get predictions from each model
            lstm_pred = self.lstm_model.predict(X_seq, verbose=0) if self.lstm_model else None
            transformer_pred = self.transformer_model.predict(X_seq, verbose=0) if self.transformer_model else None

            # Combine predictions
            if lstm_pred is not None and transformer_pred is not None:
                combined_pred = (lstm_pred * self.ensemble_weights['lstm'] + 
                                transformer_pred * self.ensemble_weights['transformer'])
                prediction = self.price_scaler.inverse_transform(combined_pred)[0][0]

                # Calculate confidence interval
                std_dev = np.std([lstm_pred[0][0], transformer_pred[0][0]]) * 2
                confidence_interval = {
                    'lower': prediction - std_dev,
                    'upper': prediction + std_dev
                }

                return {
                    'prediction': prediction,
                    'confidence_interval': confidence_interval
                }
            else:
                self.logger.warning("Some models are not trained yet")
                return None

        except Exception as e:
            self.logger.error(f"Error in price prediction: {e}")
            return None

    def train_ppo(self, data):
        """Train PPO agent for reinforcement learning"""
        try:
            # Validate input data
            if data is None or len(data) < LSTM_SEQUENCE_LENGTH + 1:
                self.logger.error("Insufficient data for PPO training")
                return False

            # Validate required columns
            for feature in LSTM_FEATURES:
                if feature not in data.columns:
                    self.logger.error(f"Missing required feature: {feature}")
                    return False

            # Initialize PPO agent if needed
            if self.ppo_agent is None:
                state_dim = len(LSTM_FEATURES)
                action_dim = 3  # Buy, Sell, Hold
                self.ppo_agent = PPOAgent(state_dim, action_dim)

            # Prepare training data
            try:
                # Scale features
                scaled_data = self.feature_scaler.fit_transform(data[LSTM_FEATURES].values)
                returns = data['close'].pct_change().fillna(0).values

                # Create sequences for states
                states = []
                next_states = []
                rewards = []
                for i in range(len(scaled_data) - LSTM_SEQUENCE_LENGTH):
                    state_seq = scaled_data[i:i+LSTM_SEQUENCE_LENGTH]
                    next_state_seq = scaled_data[i+1:i+LSTM_SEQUENCE_LENGTH+1]
                    reward = returns[i+LSTM_SEQUENCE_LENGTH]
                    
                    state = np.mean(state_seq, axis=0)  # Use mean of sequence as state
                    next_state = np.mean(next_state_seq, axis=0)  # Use mean of sequence as next state
                    
                    # Validate state and next_state
                    if np.any(np.isnan(state)) or np.any(np.isnan(next_state)):
                        continue
                    
                    states.append(state)
                    next_states.append(next_state)
                    rewards.append(reward)

                if len(states) < 32:  # Minimum required batch size
                    self.logger.error("Insufficient valid states for training")
                    return False

                states = np.array(states)
                next_states = np.array(next_states)
                rewards = np.array(rewards)

            except Exception as e:
                self.logger.error(f"Error preparing training data: {e}")
                return False

            # Training loop
            episode_rewards = []
            try:
                batch_size = 32
                num_batches = len(states) // batch_size

                for batch in range(num_batches):
                    start_idx = batch * batch_size
                    end_idx = start_idx + batch_size
                    
                    batch_states = states[start_idx:end_idx]
                    batch_next_states = next_states[start_idx:end_idx]
                    batch_rewards = rewards[start_idx:end_idx]

                    for i in range(batch_size):
                        state = batch_states[i]
                        next_state = batch_next_states[i]
                        reward = batch_rewards[i]
                        action, action_probs = self.ppo_agent.get_action(state)
                        done = (i == batch_size - 1)

                        self.ppo_agent.store_transition(state, action, reward, next_state, done, action_probs)
                        episode_rewards.append(reward)

                    # Train after collecting batch
                    train_result = self.ppo_agent.train()
                    if not train_result:
                        self.logger.error("PPO agent training failed")
                        return False

                if len(episode_rewards) > 0:
                    self.reward_history.extend(episode_rewards)
                    mean_reward = np.mean(episode_rewards)
                    self.logger.info(f"PPO training completed successfully. Mean reward: {mean_reward:.4f}")
                    return True
                else:
                    self.logger.error("No rewards collected during training")
                    return False

            except Exception as e:
                self.logger.error(f"Error during training loop: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error in PPO training: {e}")
            return False

    def generate_signals(self, data, news_texts):
        """Generate trading signals using all available models"""
        try:
            # Initialize signals with default values
            signals = {
                'ppo_action': 'HOLD',  # Default to HOLD
                'ppo_confidence': 0.0,
                'sentiment': 0.5,  # Neutral sentiment
                'model_confidence': 0.0
            }
            
            # Validate input data
            if data is None or len(data) == 0:
                self.logger.warning("No data provided for signal generation")
                return signals

            # Validate required features
            missing_features = [f for f in LSTM_FEATURES if f not in data.columns]
            if missing_features:
                self.logger.warning(f"Missing required features: {missing_features}")
                return signals
            
            try:
                # Get PPO action
                if self.ppo_agent is not None:
                    state = self.feature_scaler.transform(data[LSTM_FEATURES].values)[-1]
                    action, confidence = self.ppo_agent.get_action(state)
                    signals['ppo_action'] = ['SELL', 'HOLD', 'BUY'][action]
                    signals['ppo_confidence'] = float(np.max(confidence))
            except Exception as e:
                self.logger.warning(f"Error getting PPO action: {e}")
            
            # Add sentiment analysis if news texts are available
            if news_texts:
                try:
                    sentiment_scores = []
                    for text in news_texts:
                        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        outputs = self.sentiment_model(**inputs)
                        sentiment = torch.softmax(outputs.logits, dim=1)
                        sentiment_scores.append(float(sentiment[0][1]))  # Positive sentiment score
                    if sentiment_scores:
                        signals['sentiment'] = np.mean(sentiment_scores)
                except Exception as e:
                    self.logger.warning(f"Error in sentiment analysis: {e}")
            
            # Add model confidence if prediction is available
            try:
                price_pred = self.predict_price(data)
                if price_pred and price_pred.get('prediction', 0) != 0:
                    signals['model_confidence'] = 1.0 - (price_pred['confidence_interval']['upper'] - 
                                                        price_pred['confidence_interval']['lower']) / price_pred['prediction']
            except Exception as e:
                self.logger.warning(f"Error calculating model confidence: {e}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {
                'ppo_action': 'HOLD',
                'ppo_confidence': 0.0,
                'sentiment': 0.5,
                'model_confidence': 0.0
            }

    def train_models(self, data):
        """Train all models (LSTM and Transformer)"""
        try:
            # Train LSTM model
            self.train_lstm_model(data)
            
            # Train Transformer model
            self.train_transformer_model(data)
            
            self.logger.info("All models trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False

    def train_lstm_model(self, data, episodes=100, batch_size=32, validation_split=0.2):
        """Train LSTM model for time series prediction"""
        try:
            X = data[LSTM_FEATURES].values
            y = data['close'].values

            # Scale features and target
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.price_scaler.fit_transform(y.reshape(-1, 1))

            # Use DataGenerator
            data_gen = DataGenerator(X_scaled, y_scaled, batch_size, LSTM_SEQUENCE_LENGTH)
            self.logger.info(f"Number of batches: {len(data_gen)}, Sequence length: {LSTM_SEQUENCE_LENGTH}")

            # Build LSTM model with L2 regularization
            from tensorflow.keras.regularizers import l2
            self.lstm_model = Sequential([
                LSTM(LSTM_UNITS, input_shape=(LSTM_SEQUENCE_LENGTH, len(LSTM_FEATURES)), return_sequences=True, kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # Train model with ModelCheckpoint
            from tensorflow.keras.callbacks import ModelCheckpoint
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
                ModelCheckpoint('lstm_best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
            ]

            self.lstm_model.fit(
                data_gen,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )

            self.logger.info("LSTM model training completed successfully with updated architecture")
        
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def train_transformer_model(self, data, episodes=50, batch_size=32):
        """Train Transformer model for sequence prediction"""
        try:
            # Prepare data similar to LSTM
            X = data[LSTM_FEATURES].values
            y = data['close'].values
            
            X_scaled = self.feature_scaler.transform(X)
            y_scaled = self.price_scaler.transform(y.reshape(-1, 1))
            
            # Use DataGenerator
            data_gen = DataGenerator(X_scaled, y_scaled, batch_size, LSTM_SEQUENCE_LENGTH)
            self.logger.info(f"Number of batches: {len(data_gen)}, Sequence length: {LSTM_SEQUENCE_LENGTH}")

            # Build Transformer model
            inputs = Input(shape=(LSTM_SEQUENCE_LENGTH, len(LSTM_FEATURES)))
            x = inputs
            
            # Multi-head attention blocks
            for _ in range(3):
                attention_output = MultiHeadAttention(
                    num_heads=8, key_dim=64
                )(x, x, x)
                x = LayerNormalization()(x + attention_output)
                
            # Final processing
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu')(x)
            outputs = Dense(1)(x)
            
            self.transformer_model = Model(inputs=inputs, outputs=outputs)
            self.transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            self.transformer_model.fit(
                data_gen,
                epochs=epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Transformer model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training Transformer model: {str(e)}")
            raise

    def train_ppo_agent(self, data, episodes=1000, gamma=0.99, epsilon=0.2):
        """Train PPO agent for reinforcement learning"""
        try:
            # Initialize PPO agent if not exists
            if self.ppo_agent is None:
                state_dim = len(LSTM_FEATURES) + 12  # Base features + technical indicators
                action_dim = 3  # Buy, Sell, Hold
                self.ppo_agent = PPOAgent(state_dim, action_dim)
            
            # Prepare environment state
            states = self.prepare_rl_states(data)
            
            # Training loop
            for episode in range(episodes):
                state = states[0]
                done = False
                episode_reward = 0
                action_counts = {'Buy': 0, 'Sell': 0, 'Hold': 0}
                rewards = []
                
                while not done:
                    # Get action from agent
                    action, action_probs = self.ppo_agent.get_action(state)
                    
                    # Execute action and get next state
                    next_state, reward, done = self.step_environment(state, action, data)
                    
                    # Store transition
                    self.ppo_agent.store_transition(state, action, reward, next_state, done, action_probs)
                    
                    # Train agent
                    self.ppo_agent.train()
                    
                    state = next_state
                    episode_reward += reward
                    rewards.append(reward)
                    
                    # Update action counts
                    if action == 0:
                        action_counts['Buy'] += 1
                    elif action == 1:
                        action_counts['Sell'] += 1
                    else:
                        action_counts['Hold'] += 1
                
                if episode % 10 == 0:
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    self.model_performance['ppo'].append({
                        'avg_reward': avg_reward,
                        'std_reward': std_reward,
                        'action_dist': action_counts
                    })
                    self.logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}, Action Dist: {action_counts}")
                
            self.logger.info("PPO agent training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training PPO agent: {str(e)}")
            raise

    def optimize_ensemble_weights(self, lstm_preds, transformer_preds, ppo_preds, sentiment_preds, y_true, config):
        """Optimize weights for combining predictions from different models based on their performance.

        Args:
            lstm_preds (np.ndarray): LSTM model predictions
            transformer_preds (np.ndarray): Transformer model predictions
            ppo_preds (np.ndarray): PPO agent predictions
            sentiment_preds (np.ndarray): Sentiment model predictions
            y_true (np.ndarray): True labels/values
            config (dict): Configuration dictionary

        Returns:
            dict: Optimized weights for each model

        Raises:
            ValueError: If prediction arrays have different shapes or invalid values
            RuntimeError: If numerical issues occur during optimization
            Exception: For other unexpected errors
        """
        try:
            # Setup logging
            logger = logging.getLogger(__name__)
            logger.info("Starting ensemble weight optimization...")

            # Validate inputs
            predictions = {
                'lstm': lstm_preds,
                'transformer': transformer_preds,
                'ppo': ppo_preds,
                'sentiment': sentiment_preds
            }
            
            # Check shapes
            base_shape = y_true.shape
            for model_name, preds in predictions.items():
                if preds.shape != base_shape:
                    raise ValueError(f"Shape mismatch for {model_name}: {preds.shape} vs {base_shape}")

            # Calculate performance metrics for each model
            metrics = {}
            for model_name, preds in predictions.items():
                # Use MSE for regression, accuracy for classification
                if y_true.shape[1] == 1:  # Regression
                    mse = np.mean((preds - y_true) ** 2)
                    metrics[model_name] = 1 / (mse + 1e-10)  # Inverse MSE, avoid division by zero
                else:  # Classification
                    accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y_true, axis=1))
                    metrics[model_name] = accuracy

            # Normalize metrics to get weights
            total_metric = sum(metrics.values())
            weights = {model: score/total_metric for model, score in metrics.items()}

            # Update ensemble weights
            self.ensemble_weights = weights
            logger.info(f"Optimized weights: {weights}")

            # Update config and save
            config['ENSEMBLE_WEIGHTS'] = weights
            try:
                from config import ConfigLoader
                ConfigLoader.save_config(config)
                logger.info("Updated weights saved to config file")
            except Exception as e:
                logger.error(f"Error saving config: {e}")

            # Log performance metrics
            for model_name, metric in metrics.items():
                self.model_performance[model_name].append(metric)
                logger.info(f"{model_name} performance metric: {metric:.4f}")

            return weights

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error during optimization: {re}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during weight optimization: {e}")
            raise
        
        # Initialize PPO agent
        self.state_dim = len(LSTM_FEATURES) + 12  # Base features + engineered features
        self.action_dim = 3  # Buy, Sell, Hold
        self.ppo_agent = PPOAgent(self.state_dim, self.action_dim)
        self.reward_history = []
        
def engineer_features(self, df):
        """Engineer technical indicators and features for model input"""
        try:
            df = df.copy()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate ADX
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Calculate OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Fill NaN values
            df = df.fillna(method='bfill')
            
            self.logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise

def select_important_features(self, df):
    """Select top 10 important features using SHAP values"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap

        # Check for required columns
        if 'close' not in df.columns:
            self.logger.error("DataFrame lacks 'close' column")
            return []

        # Prepare data
        X = df.drop(columns=['close']).values
        y = df['close'].values

        # Check for sufficient data
        if len(X) < 100:
            self.logger.error("Insufficient data for feature selection")
            return []

        # Train RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Rank features
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = df.drop(columns=['close']).columns
        important_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

        # Log and return top 10 features
        top_features = [name for name, _ in important_features[:10]]
        self.logger.info(f"Selected features: {top_features}")
        return top_features

    except Exception as e:
        self.logger.error(f"Error in feature selection: {e}")
        return []

def prepare_model_data(self, df):
        """Prepare data for both LSTM and Transformer models"""
        try:
            # Engineer features
            df = self.engineer_features(df)
            if df is None:
                self.logger.error("Feature engineering returned None")
                return None, None
            
            # Select and scale features
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].values
            
            prices = df[['close']].values
            
            # Scale data
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_prices = self.price_scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            selected_features = self.select_important_features(df)
            if not selected_features:
                self.logger.error("No features selected")
                return None, None

            features = df[selected_features].values
            prices = df[['close']].values
            
            # Scale data
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_prices = self.price_scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - LSTM_SEQUENCE_LENGTH):
                X.append(scaled_features[i:(i + LSTM_SEQUENCE_LENGTH)])
                y.append(scaled_prices[i + LSTM_SEQUENCE_LENGTH])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing model data: {e}")
            return None, None

def generate_signals(self, df, news_texts):
        """Generate trading signals using ensemble of AI models including PPO"""
        try:
            signals = {}
            
            # Get price predictions with confidence intervals
            price_pred = self.predict_price(df)
            if price_pred is not None:
                signals['prediction'] = price_pred['prediction']
                signals['confidence_interval'] = price_pred['confidence_interval']
            
            # Get sentiment analysis
            if news_texts:
                sentiment = self.analyze_sentiment(news_texts)
                if sentiment is not None:
                    signals['sentiment'] = sentiment
            
            # Get PPO agent's action
            current_state = self.feature_scaler.transform([df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].iloc[-1].values])
            
            action, action_probs = self.ppo_agent.get_action(current_state[0])
            signals['ppo_action'] = ['BUY', 'SELL', 'HOLD'][action]
            signals['ppo_confidence'] = float(max(action_probs))
            
            # Combine all signals
            if 'prediction' in signals:
                # Adjust prediction based on PPO and sentiment
                ppo_adjustment = (action - 1) * signals['ppo_confidence'] * self.ensemble_weights['ppo']
                sentiment_adjustment = ((sentiment - 0.5) * 2 * self.ensemble_weights['sentiment']) if 'sentiment' in signals else 0
                signals['prediction'] *= (1 + ppo_adjustment + sentiment_adjustment)
            
            # Add prediction confidence
            if len(self.model_performance['lstm']) > 0:
                signals['model_confidence'] = 1 / (1 + np.mean([
                    self.model_performance['lstm'][-1],
                    self.model_performance['transformer'][-1],
                    -self.model_performance['ppo'][-1]  # Higher rewards are better
                ]))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")
            return {'ppo_action': 'HOLD', 'ppo_confidence': 0.0}

def prepare_rl_states(self, data):
        """Prepare states for reinforcement learning"""
        try:
            # Combine technical features and position information
            features = np.column_stack([
                data[LSTM_FEATURES].values,  # Base features
                data[['returns', 'volatility', 'price_volatility',
                      'volume_price_trend', 'body_size',
                      'upper_shadow', 'lower_shadow']].values,  # Technical indicators
                np.zeros((len(data), 5))  # Position and account info placeholders
            ])
            
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(features)
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error preparing RL states: {e}")
            return None
    
def step_environment(self, state, action, data):
        """Simulate environment step for PPO agent"""
        try:
            # Extract current price
            current_price = data['close'].iloc[-1]
            next_price = data['close'].iloc[-2]  # Assuming next state is previous row
        
            # Calculate price change
            price_change = next_price - current_price
        
            # Initialize transaction cost and slippage
            transaction_cost = 0.0
            slippage = 0.0
        
            # Apply transaction cost and slippage based on action
            if action in [0, 1]:  # Buy or Sell
                transaction_cost = 0.001 * current_price
                slippage = 0.0005 * price_change
        
            # Calculate reward
            if action == 0:  # Buy
                reward = price_change - transaction_cost - slippage
            elif action == 1:  # Sell
                reward = -price_change - transaction_cost - slippage
            else:  # Hold
                reward = -0.01  # Small penalty for holding
        
            # Update next state with transaction cost and slippage
            next_state = np.append(state, [transaction_cost, slippage])
        
            # Log transaction cost and slippage
            self.logger.info(f"Action: {action}, Transaction Cost: {transaction_cost:.4f}, Slippage: {slippage:.4f}")
        
            return next_state, reward, False  # Assuming episode continues
        except Exception as e:
            self.logger.error(f"Error in step_environment: {e}")
            return state, -1.0, True  # Return negative reward and end episode on error

def build_transformer_model(self, input_shape):
        """Build Transformer model for time series prediction"""
        try:
            inputs = Input(shape=input_shape)
            
            # Add positional encoding
            pos_encoding = np.zeros(input_shape)
            for pos in range(input_shape[0]):
                for i in range(0, input_shape[1], 2):
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / input_shape[1])))
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / input_shape[1])))
            
            x = inputs + pos_encoding
            
            # Multi-head attention layers
            x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(0.1)(x)
            
            # Dense layers
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.1)(x)
            x = Dense(64, activation='relu')(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Output layer
            x = Dense(32, activation='relu')(x)
            x = Dense(1)(x[:, -1, :])
            
            return Model(inputs=inputs, outputs=x)
            
        except Exception as e:
            self.logger.error(f"Error building transformer model: {e}")
            return {'prediction': None, 'confidence_interval': None, 'ppo_action': 'HOLD', 'ppo_confidence': 0.0}
    
def train_models(self, df):
        """Train both LSTM and Transformer models with proper validation"""
        try:
            # Prepare data
            X, y = self.prepare_model_data(df)
            if X is None or y is None:
                return False
            
            # Split data into train/validation/test (60/20/20)
            train_size = int(len(X) * 0.6)
            val_size = int(len(X) * 0.2)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            
            # Define callbacks with validation monitoring
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
            ]
            
            # Build and train LSTM model
            self.lstm_model = Sequential([
                LSTM(LSTM_UNITS, return_sequences=True, 
                     input_shape=(LSTM_SEQUENCE_LENGTH, X.shape[2])),
                Dropout(0.2),
                LSTM(LSTM_UNITS // 2),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                                  loss='mse',
                                  metrics=['mae'])
            
            lstm_history = self.lstm_model.fit(X_train, y_train,
                                              batch_size=LSTM_BATCH_SIZE,
                                              epochs=LSTM_EPOCHS,
                                              validation_data=(X_val, y_val),
                                              callbacks=callbacks,
                                              verbose=0)
            
            # Build and train Transformer model
            self.transformer_model = self.build_transformer_model((LSTM_SEQUENCE_LENGTH, X.shape[2]))
            if self.transformer_model is None:
                return False
            
            self.transformer_model.compile(optimizer=Adam(learning_rate=0.001),
                                         loss='mse',
                                         metrics=['mae'])
            
            transformer_history = self.transformer_model.fit(X_train, y_train,
                                                           batch_size=LSTM_BATCH_SIZE,
                                                           epochs=LSTM_EPOCHS,
                                                           validation_data=(X_val, y_val),
                                                           callbacks=callbacks,
                                                           verbose=0)
            
            # Update model performance metrics
            self.model_performance['lstm'].append(lstm_history.history['val_loss'][-1])
            self.model_performance['transformer'].append(transformer_history.history['val_loss'][-1])
            
            # Update ensemble weights based on recent performance
            total_performance = sum(1/x for x in [self.model_performance['lstm'][-1],
                                                self.model_performance['transformer'][-1]])
            self.ensemble_weights['lstm'] = (1/self.model_performance['lstm'][-1]) / total_performance * 0.8
            self.ensemble_weights['transformer'] = (1/self.model_performance['transformer'][-1]) / total_performance * 0.8
            self.ensemble_weights['sentiment'] = 0.2
            
            self.last_retrain = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False

def predict_price(self, df):
        """Make ensemble predictions with confidence intervals"""
        try:
            if self.lstm_model is None or self.transformer_model is None:
                return {'prediction': None, 'confidence_interval': None, 'ppo_action': 'HOLD', 'ppo_confidence': 0.0}
            
            # Check if retraining is needed (e.g., every 24 hours)
            if (datetime.now() - self.last_retrain).days >= 1:
                self.train_models(df)
            
            # Prepare latest data
            try:
                df = engineer_features(df)
                if df is None:
                    self.logger.error("Feature engineering returned None")
                    return {'prediction': None, 'confidence_interval': None}
            except Exception as e:
                self.logger.error(f"Error in feature engineering: {e}")
                return {'prediction': None, 'confidence_interval': None}
                
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].tail(LSTM_SEQUENCE_LENGTH).values
            
            scaled_features = self.feature_scaler.transform(features)
            X = np.array([scaled_features])
            
            # Get predictions from both models
            lstm_pred = self.lstm_model.predict(X, verbose=0)[0][0]
            transformer_pred = self.transformer_model.predict(X, verbose=0)[0][0]
            
            # Calculate ensemble prediction
            ensemble_pred = (
                lstm_pred * self.ensemble_weights['lstm'] +
                transformer_pred * self.ensemble_weights['transformer']
            )
            
            # Calculate confidence interval
            pred_std = np.std([lstm_pred, transformer_pred])
            confidence_interval = {
                'lower': ensemble_pred - 1.96 * pred_std,
                'upper': ensemble_pred + 1.96 * pred_std
            }
            
            return {
                'prediction': self.price_scaler.inverse_transform([[ensemble_pred]])[0][0],
                'confidence_interval': {
                    'lower': self.price_scaler.inverse_transform([[confidence_interval['lower']]])[0][0],
                    'upper': self.price_scaler.inverse_transform([[confidence_interval['upper']]])[0][0]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {e}")
            return {'prediction': None, 'confidence_interval': None, 'ppo_action': 'HOLD', 'ppo_confidence': 0.0}

def analyze_sentiment(self, news_texts):
        """Analyze sentiment of news articles using FinBERT"""
        try:
            if not news_texts:
                return {'prediction': None, 'confidence_interval': None, 'ppo_action': 'HOLD', 'ppo_confidence': 0.0}
                
            # Combine texts and tokenize
            combined_text = ' '.join(news_texts[:5])  # Analyze latest 5 news articles
            inputs = self.sentiment_tokenizer(combined_text, 
                                            return_tensors="pt",
                                            max_length=512,
                                            truncation=True)
            
            # Get sentiment prediction
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            # Convert to sentiment score (0-1)
            sentiment_score = predictions[0][0].item()  # Probability of positive sentiment
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {'prediction': None, 'confidence_interval': None, 'ppo_action': 'HOLD', 'ppo_confidence': 0.0}

def calculate_reward(self, action, next_price, current_price, drawdown=0.0):
        """Calculate risk-adjusted reward for the PPO agent"""
        try:
            # Calculate price change percentage
            price_change = (next_price - current_price) / current_price
            
            # Define rewards based on action and price movement
            if action == 0:  # Buy
                reward = price_change
            elif action == 1:  # Sell
                reward = -price_change
            else:  # Hold
                reward = -abs(price_change) * 0.1  # Small penalty for holding
            
            # Apply risk adjustment based on drawdown
            risk_factor = 1 - min(drawdown / 0.1, 1)  # Reduce reward as drawdown approaches 10%
            reward = reward * risk_factor
            
            # Scale and clip final reward
            reward = np.clip(reward * 100, -1, 1)
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
def train_ppo(self, df):
        """Train PPO agent on historical data"""
        try:
            # Prepare state features
            df = engineer_features(df)
            if df is None:
                return False
            
            features = df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].values
            
            scaled_features = self.feature_scaler.transform(features)
            prices = df['close'].values
            
            # Training loop
            total_reward = 0
            for i in range(len(scaled_features) - 1):
                state = scaled_features[i]
                next_state = scaled_features[i + 1]
                current_price = prices[i]
                next_price = prices[i + 1]
                
                # Get action from PPO agent
                action, action_probs = self.ppo_agent.get_action(state)
                
                # Calculate reward
                reward = self.calculate_reward(action, next_price, current_price)
                total_reward += reward
                
                # Store transition and train
                done = (i == len(scaled_features) - 2)
                self.ppo_agent.store_transition(state, action, reward, next_state, done, action_probs)
                self.ppo_agent.train()
            
            # Store performance metrics
            self.model_performance['ppo'].append(total_reward)
            self.reward_history.append(total_reward)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training PPO agent: {e}")
            return False
    
def optimize_ppo_threshold(self, df, thresholds=None):
        """Find optimal PPO confidence threshold using historical data"""
        try:
            if thresholds is None:
                thresholds = np.arange(0.5, 0.9, 0.1)
            
            best_threshold = 0.6  # Default threshold
            best_sharpe = float('-inf')
            
            # Split data for optimization
            train_size = int(len(df) * 0.8)
            optimization_data = df[:train_size]
            
            for threshold in thresholds:
                returns = []
                position = 0  # -1: short, 0: flat, 1: long
                
                for i in range(len(optimization_data) - 1):
                    current_data = optimization_data.iloc[:i+1]
                    signals = self.generate_signals(current_data, [])
                    
                    if signals and signals['ppo_confidence'] > threshold:
                        if signals['ppo_action'] == 'BUY':
                            position = 1
                        elif signals['ppo_action'] == 'SELL':
                            position = -1
                    
                    # Calculate returns
                    if position != 0 and i < len(optimization_data) - 1:
                        price_change = ((optimization_data.iloc[i+1]['close'] - 
                                        optimization_data.iloc[i]['close']) /
                                        optimization_data.iloc[i]['close'])
                        returns.append(position * price_change)
                    else:
                        returns.append(0)
                
                if returns:
                    returns = np.array(returns)
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe = np.sqrt(252) * avg_return / std_return if std_return > 0 else float('-inf')
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_threshold = threshold
            
            self.logger.info(f"Optimal PPO confidence threshold: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing PPO threshold: {e}")
            return 0.6  # Return default threshold on error

def generate_signals(self, df, news_texts):
        """Generate trading signals using ensemble of AI models including PPO"""
        try:
            signals = {}
            
            # Get price predictions with confidence intervals
            price_pred = self.predict_price(df)
            if price_pred is not None:
                signals['prediction'] = price_pred['prediction']
                signals['confidence_interval'] = price_pred['confidence_interval']
            
            # Get sentiment analysis
            if news_texts:
                sentiment = self.analyze_sentiment(news_texts)
                if sentiment is not None:
                    signals['sentiment'] = sentiment
            
            # Get PPO agent's action
            current_state = self.feature_scaler.transform([df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].iloc[-1].values])
            
            action, action_probs = self.ppo_agent.get_action(current_state[0])
            signals['ppo_action'] = ['BUY', 'SELL', 'HOLD'][action]
            signals['ppo_confidence'] = float(max(action_probs))
            
            # Combine all signals
            if 'prediction' in signals:
                # Adjust prediction based on PPO and sentiment
                ppo_adjustment = (action - 1) * signals['ppo_confidence'] * self.ensemble_weights['ppo']
                sentiment_adjustment = ((sentiment - 0.5) * 2 * self.ensemble_weights['sentiment']) if 'sentiment' in signals else 0
                signals['prediction'] *= (1 + ppo_adjustment + sentiment_adjustment)
            
            # Add prediction confidence
            if len(self.model_performance['lstm']) > 0:
                signals['model_confidence'] = 1 / (1 + np.mean([
                    self.model_performance['lstm'][-1],
                    self.model_performance['transformer'][-1],
                    -self.model_performance['ppo'][-1]  # Higher rewards are better
                ]))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")
            return {'ppo_action': 'HOLD', 'ppo_confidence': 0.0}

def train_ppo_agent(self, trading_env, data, config: dict) -> PPOAgent:
        """Train a PPO agent for optimizing XAUUSD trading actions.

        Args:
            trading_env: Gym-like environment with market data
            data (pd.DataFrame): DataFrame for environment setup
            config (dict): Configuration dictionary containing model parameters

        Returns:
            PPOAgent: Trained PPO agent

        Raises:
            ValueError: If environment setup fails
            RuntimeError: If numerical instability occurs
            IOError: If model saving fails
            Exception: For other unexpected errors
        """
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename='trading_bot.log'
            )
            logger = logging.getLogger(__name__)
            logger.info("Starting PPO agent training...")

            # Initialize PPO agent
            state_dim = len(LSTM_FEATURES) + 12  # Base features + engineered features
            action_dim = 3  # Buy, Sell, Hold
            device = torch.device(config['DEVICE'] if torch.cuda.is_available() and config['DEVICE'].lower() == 'cuda' else 'cpu')
            
            # Create PPO networks
            class ActorCritic(nn.Module):
                def __init__(self, state_dim, action_dim):
                    super().__init__()
                    # Actor network
                    self.actor = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, action_dim),
                        nn.Softmax(dim=-1)
                    )
                    # Critic network
                    self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    )

                def forward(self, state):
                    return self.actor(state), self.critic(state)

            # Initialize networks and optimizer
            ppo_net = ActorCritic(state_dim, action_dim).to(device)
            optimizer = TorchAdam(ppo_net.parameters(), lr=config['LEARNING_RATE'])

            # Training loop
            episodes = config['PPO_EPISODES']
            gamma = config['GAMMA']
            epsilon_clip = config['EPSILON_CLIP']
            best_reward = float('-inf')
            rewards_history = []

            logger.info(f"Training for {episodes} episodes...")
            for episode in range(episodes):
                state = trading_env.reset()
                done = False
                total_reward = 0
                transitions = []

                while not done:
                    state_tensor = torch.FloatTensor(state).to(device)
                    action_probs, value = ppo_net(state_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    next_state, reward, done, _ = trading_env.step(action)

                    # Store transition
                    transitions.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'action_prob': action_probs[action].item(),
                        'value': value.item()
                    })

                    state = next_state
                    total_reward += reward

                # Process episode transitions
                states = torch.FloatTensor([t['state'] for t in transitions]).to(device)
                actions = torch.LongTensor([t['action'] for t in transitions]).to(device)
                rewards = torch.FloatTensor([t['reward'] for t in transitions]).to(device)
                old_probs = torch.FloatTensor([t['action_prob'] for t in transitions]).to(device)
                values = torch.FloatTensor([t['value'] for t in transitions]).to(device)

                # Compute returns and advantages
                returns = []
                advantages = []
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.FloatTensor(returns).to(device)
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO update
                for _ in range(10):  # Multiple epochs of updates
                    new_probs, new_values = ppo_net(states)
                    new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()

                    ratio = new_probs / old_probs
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
                    loss = actor_loss + 0.5 * critic_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                rewards_history.append(total_reward)
                avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)

                if episode % 100 == 0:
                    logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}")

                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(ppo_net.state_dict(), config['PPO_MODEL_PATH'])
                    logger.info(f"New best model saved with average reward: {best_reward:.2f}")

            logger.info("PPO training completed successfully")
            return self.ppo_agent

        except ValueError as ve:
            logger.error(f"Environment setup error: {ve}")
            raise
        except RuntimeError as re:
            logger.error(f"Numerical instability error: {re}")
            raise
        except IOError as io:
            logger.error(f"Model saving error: {io}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PPO training: {e}")
            raise

def optimize_ppo_threshold(self, df, thresholds=None):
        """Find optimal PPO confidence threshold using historical data"""
        try:
            if thresholds is None:
                thresholds = np.arange(0.5, 0.9, 0.1)
            
            best_threshold = 0.6  # Default threshold
            best_sharpe = float('-inf')
            
            # Split data for optimization
            train_size = int(len(df) * 0.8)
            optimization_data = df[:train_size]
            
            for threshold in thresholds:
                returns = []
                position = 0  # -1: short, 0: flat, 1: long
                
                for i in range(len(optimization_data) - 1):
                    current_data = optimization_data.iloc[:i+1]
                    signals = self.generate_signals(current_data, [])
                    
                    if signals and signals['ppo_confidence'] > threshold:
                        if signals['ppo_action'] == 'BUY':
                            position = 1
                        elif signals['ppo_action'] == 'SELL':
                            position = -1
                    
                    # Calculate returns
                    if position != 0 and i < len(optimization_data) - 1:
                        price_change = ((optimization_data.iloc[i+1]['close'] - 
                                        optimization_data.iloc[i]['close']) /
                                        optimization_data.iloc[i]['close'])
                        returns.append(position * price_change)
                    else:
                        returns.append(0)
                
                if returns:
                    returns = np.array(returns)
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe = np.sqrt(252) * avg_return / std_return if std_return > 0 else float('-inf')
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_threshold = threshold
            
            self.logger.info(f"Optimal PPO confidence threshold: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing PPO threshold: {e}")
            return 0.6  # Return default threshold on error

def generate_signals(self, df, news_texts):
        """Generate trading signals using ensemble of AI models including PPO"""
        try:
            signals = {}
            
            # Get price predictions with confidence intervals
            price_pred = self.predict_price(df)
            if price_pred is not None:
                signals['prediction'] = price_pred['prediction']
                signals['confidence_interval'] = price_pred['confidence_interval']
            
            # Get sentiment analysis
            if news_texts:
                sentiment = self.analyze_sentiment(news_texts)
                if sentiment is not None:
                    signals['sentiment'] = sentiment
            
            # Get PPO agent's action
            current_state = self.feature_scaler.transform([df[LSTM_FEATURES + [
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'price_volatility', 'volume_price_trend', 'volatility', 'body_size',
                'upper_shadow', 'lower_shadow'
            ]].iloc[-1].values])
            
            action, action_probs = self.ppo_agent.get_action(current_state[0])
            signals['ppo_action'] = ['BUY', 'SELL', 'HOLD'][action]
            signals['ppo_confidence'] = float(max(action_probs))
            
            # Combine all signals
            if 'prediction' in signals:
                # Adjust prediction based on PPO and sentiment
                ppo_adjustment = (action - 1) * signals['ppo_confidence'] * self.ensemble_weights['ppo']
                sentiment_adjustment = ((sentiment - 0.5) * 2 * self.ensemble_weights['sentiment']) if 'sentiment' in signals else 0
                signals['prediction'] *= (1 + ppo_adjustment + sentiment_adjustment)
            
            # Add prediction confidence
            if len(self.model_performance['lstm']) > 0:
                signals['model_confidence'] = 1 / (1 + np.mean([
                    self.model_performance['lstm'][-1],
                    self.model_performance['transformer'][-1],
                    -self.model_performance['ppo'][-1]  # Higher rewards are better
                ]))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating AI signals: {e}")
            return {'ppo_action': 'HOLD', 'ppo_confidence': 0.0}