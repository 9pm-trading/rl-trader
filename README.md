# Forex Trading AI System

This project implements an AI-driven forex trading system using reinforcement learning techniques. The system consists of three main components: a linear model for decision making, an environment simulator, and a database interface.

## Project Structure
```
├── DATA_PREP
│   ├── eur m1.ipynb
│   ├── eur_h1.ipynb
│   └── mapping h1 and m1.ipynb
├── MT4
│   ├── H1Bar v2.1.debug.mq4
│   ├── H1Bar v2.1.mq4
│   └── straddle 4.1.slave.mq4
├── README.md
└── RL_SYSTEM
├── core.py
├── db.py
└── linear_fx_trader_2_16.py
```

## Components

### RL_SYSTEM

#### 1. linear_fx_trader_2_16.py

This file contains the main trading logic and reinforcement learning model.

Key features:
- Implements a Linear Model for Q-learning
- Defines a MultiStockEnv class that simulates the trading environment
- Implements a DQNAgent class for decision making
- Provides functions for training and testing the model
- Includes real-time trading capabilities

#### 2. core.py

This file defines the core functionality of the trading system.

Key features:
- Implements the PairEnv class, which simulates a more detailed forex trading environment
- Defines the LinearModel and DQNAgent classes
- Provides utility functions for data preprocessing and model evaluation
- Implements a sophisticated trading strategy with multiple order levels and risk management

#### 3. db.py

This file handles database operations for storing and retrieving trading data.

Key features:
- Provides functions to interact with a MySQL database
- Handles operations for retrieving market data
- Manages order information and system status

## Setup and Usage

1. Ensure you have Python 3.x installed along with the required libraries (numpy, pandas, matplotlib, scikit-learn, mysql-connector).

2. Set up a MySQL database and configure the connection details in db.py.

3. Prepare your historical forex data using the notebooks in the DATA_PREP folder.

4. To train the model:
```
python RL_SYSTEM/linear_fx_trader_2_16.py -m train
```
5. To test the model:
```
python RL_SYSTEM/linear_fx_trader_2_16.py -m test
```
6. For real-time trading (ensure your database is continuously updated with live market data):
```
python RL_SYSTEM/linear_fx_trader_2_16.py -m real
```
## Data Preparation

Before training the model, it's crucial to prepare the data correctly. Use the notebooks in the DATA_PREP folder for this purpose.

### Data Processing Steps

1. Use `eur m1.ipynb` to process M1 (1-minute) data
2. Use `eur_h1.ipynb` to process H1 (1-hour) data
3. Use `mapping h1 and m1.ipynb` to merge the M1 and H1 data

### Important Notes

- The date range for the data is 2013-2020.
- Ensure that the "date" and "hour" fields are correctly formatted in both M1 and H1 datasets for proper merging.
- The final merged file should contain aligned H1 and M1 data, which is crucial for the model's training process.

## MetaTrader Integration

This system integrates with MetaTrader 4 for live data feeding and trade execution. Two custom MetaTrader 4 Expert Advisors (EAs) are provided in the MT4 folder.

### Live Data Feeder: H1Bar v2.1.mq4

This EA is responsible for feeding live market data from MetaTrader 4 into the MySQL database.

### Trade Executor: straddle 4.1.slave.mq4

This EA reads the inference results from the API and executes trades in MetaTrader 4.

### Important Notes

- The API server implementation that bridges these EAs with the MySQL database is not included in this repository.
- You will need to implement your own API server to facilitate communication between MetaTrader 4 and the MySQL database.
- Ensure that your API implementation includes proper security measures and error handling.
- Test thoroughly in a demo environment before using with real funds.