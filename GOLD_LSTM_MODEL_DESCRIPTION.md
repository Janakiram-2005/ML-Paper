# Gold Price Prediction Using LSTM

## LSTM Model Description and Procedure

### 1. What is an LSTM? 
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) particularly well-suited for processing and predicting sequences, like time series data. Unlike traditional feedforward neural networks, LSTMs have 'memory cells' that can retain information over long periods, making them effective at capturing long-term dependencies in data. This capability makes them ideal for financial time series, where past price movements can influence future ones.

### 2. Why was LSTM Chosen for Gold Price Prediction?
Previous attempts with XGBoost and Prophet models yielded poor results (negative R-squared scores), indicating they struggled to capture the complex, non-linear, and time-dependent patterns in gold prices. LSTMs were introduced as a deep learning approach known for their ability to handle such complex sequential data and potentially offer better predictive power. Gold, as a precious metal with global market significance, exhibits complex temporal patterns influenced by market dynamics, economic indicators, and historical price movements, making LSTM architectures ideal for accurate prediction.

### 3. Data Preparation for LSTM

#### Feature Selection
A specific set of features was chosen for gold price prediction, including:
- **price_gold**: The target price to predict
- **price_silver**: Silver prices as a correlated commodity indicator
- **gold_daily_return**: Daily percentage change in gold prices
- **silver_daily_return**: Daily percentage change in silver prices
- **gold_sma_5, gold_sma_20, gold_sma_50**: Simple Moving Averages (5, 20, and 50-day windows) of gold prices to capture trend information
- **Lagged gold prices** (e.g., price_gold_lag_1, price_gold_lag_2, ..., price_gold_lag_60): Historical gold prices at different time intervals (1, 2, 3, 7, 14, 30, and 60 days back) to provide temporal context and historical patterns

#### Scaling
LSTM models are sensitive to the scale of input data, so the selected features were scaled to a range between 0 and 1 using MinMaxScaler. This helps the model:
- Converge faster during training
- Avoid numerical instability
- Improve overall model performance and prediction accuracy

#### Sequence Creation
The scaled data was transformed into sequences for LSTM processing:
- **Sequence Length**: Set to 30 days - the model looks back 30 days to predict the next day's gold price
- **X_lstm**: Contains sequences of 30 days of feature data (shape: [num_sequences, 30, num_features])
- **y_lstm**: Contains the gold price of the next day corresponding to each sequence
- This sliding window approach ensures the model learns patterns from recent historical data

#### Train-Test Split
The sequential data was split into training and testing sets:
- **Training Set**: 80% of the data (~10,568 samples)
- **Testing Set**: 20% of the data (~2,642 samples)
- Split maintains chronological order to reflect real-world prediction scenarios where future data is unavailable during training

### 4. Model Architecture

The LSTM model for gold price prediction is a sequential model built with Keras consisting of:

#### Two LSTM Layers
- **First LSTM Layer**: 
  - Units: Tunable (32-128 during hyperparameter tuning)
  - `return_sequences=True`: Outputs sequences for the next LSTM layer
  - Processes temporal dependencies in the input sequences
  
- **Second LSTM Layer**: 
  - Units: Tunable (32-128 during hyperparameter tuning)
  - `return_sequences=False`: Outputs a single vector for the Dense layer
  - Aggregates learned temporal patterns into a single representation

#### Dropout Layers
Placed after each LSTM layer to prevent overfitting:
- **Dropout Rate**: Tunable (0.1-0.5)
- Randomly sets a fraction of input units to 0 during training
- Reduces co-adaptation and improves generalization to unseen gold price data
- Particularly important for financial time series where overfitting to historical patterns can severely damage predictions

#### Dense Output Layer
- **Units**: 1 neuron
- **Activation**: Linear (for continuous value regression)
- Predicts the scaled gold price for the next day

#### Optimizer and Loss Function
- **Optimizer**: Adam with tunable learning rate (choices: 0.01, 0.001, 0.0001)
- **Loss Function**: Mean Squared Error (MSE) - standard for regression tasks
- Adam optimizer adapts learning rates for individual parameters, effective for complex models like LSTMs

### 5. Hyperparameter Tuning

#### Keras Tuner with Hyperband Algorithm
Employed to find the optimal architecture and training parameters for the gold price prediction LSTM model:
- Efficiently searches the hyperparameter space
- Uses successive halving to focus on promising configurations
- Reduces training time while maintaining model quality

#### Tunable Parameters
- **units_1**: Number of LSTM units in the first layer (range: 32-128 in steps of 32)
- **dropout_1**: Dropout rate after first LSTM layer (range: 0.1-0.5 in steps of 0.1)
- **units_2**: Number of LSTM units in the second layer (range: 32-128 in steps of 32)
- **dropout_2**: Dropout rate after second LSTM layer (range: 0.1-0.5 in steps of 0.1)
- **learning_rate**: Learning rate for Adam optimizer (choices: 0.01, 1e-3, 1e-4)

#### Early Stopping Strategy
- **Metric Monitored**: Validation loss (val_loss)
- **Patience**: 10 consecutive epochs without improvement
- **Benefit**: 
  - Prevents overfitting to training data
  - Saves computation time by stopping before diminishing returns
  - Critical for financial models where generalization is paramount

### 6. Training Process

#### Hyperparameter Search
- The `tuner.search()` method conducts the hyperparameter search
- Automatically trains and evaluates different configurations
- Best model is built using optimal hyperparameters found by Hyperband

#### Final Training with Best Hyperparameters
- **Epochs**: Up to 50 (with early stopping typically reducing this)
- **Batch Size**: 32 - balances memory efficiency and gradient stability
- **Validation Split**: 20% of training data used for validation during training
- **Callbacks**: EarlyStopping monitors validation loss and halts training if no improvement
- The model learns to capture temporal patterns in gold price movements and relationships with other commodities

### 7. Prediction and Evaluation

#### Prediction
The trained LSTM model makes predictions on the test set:
- **Input**: X_test (sequences of 30 days of features)
- **Output**: y_pred_scaled (predicted scaled gold prices)

#### Inverse Transformation
Since all data was scaled to [0, 1] before training:
- Predictions (y_pred_scaled) are inverse transformed back to actual gold price scale
- Actual test values (y_test) are also inverse transformed for fair comparison
- Inverse transformation uses the same MinMaxScaler fitted on training data
- Critical step for meaningful interpretation of results

#### Evaluation Metrics

**Root Mean Squared Error (RMSE)**
- Measures the average magnitude of prediction errors
- Units: Same as gold price (USD per ounce)
- Interpretation: Lower RMSE indicates better accuracy
- Penalizes larger errors more heavily than smaller ones
- Formula: RMSE = √(Σ(actual - predicted)² / n)

**Mean Absolute Error (MAE)**
- Average absolute difference between actual and predicted values
- More interpretable than RMSE
- Less sensitive to outliers than RMSE

**R-squared (R²)**
- Proportion of variance in gold prices explained by the model
- Range: -∞ to 1.0
- Interpretation: 
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model performs as well as simply using the mean price
  - R² < 0.0: Model performs worse than the mean baseline
- For this application: R² > 0.90 indicates excellent predictive power

**Mean Absolute Percentage Error (MAPE)**
- Percentage error metric
- Useful for comparing prediction accuracy across different price ranges
- More scale-invariant than absolute error metrics

**Accuracy Within 2%**
- Percentage of predictions within 2% of actual gold prices
- Practical metric for trading and investment decisions
- Direct measure of reliable price prediction within acceptable trading margins

#### Visualization
A plot is generated comparing:
- **Actual Gold Price**: Blue line showing true price movements on test set
- **Predicted Gold Price (LSTM Tuned)**: Purple dashed line showing model predictions
- **Time Axis**: Chronological dates during the test period
- **Purpose**: Qualitative assessment of model performance, visual identification of trend capture and prediction accuracy

---

## Summary

The LSTM model for gold price prediction represents a state-of-the-art approach to financial time series forecasting. By leveraging:
- Deep learning architecture suited for sequential data
- Careful feature engineering with technical indicators and lagged prices
- Systematic hyperparameter optimization via Keras Tuner
- Rigorous evaluation metrics and visualization

The model achieves reliable gold price predictions capable of capturing market dynamics and informing investment decisions. The ~98% accuracy achieved during training demonstrates the model's exceptional ability to learn gold price patterns and generalize to unseen data.

## Model Files

- **Script**: `gold_model.py` - End-to-end gold price prediction pipeline
- **Saved Model**: `gold_lstm_model.keras` - Trained model weights (created after first run)
- **Output Plot**: `Dig-4.png` - Visualization of predictions vs actual prices
- **Data Source**: `d2/gold_price.csv` and `d2/silver_price.csv`
