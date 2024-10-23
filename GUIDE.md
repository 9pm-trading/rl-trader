Here's the high-level Mermaid flowchart:

```mermaid
graph TD
    A[Start] --> B[Initialize Environment]
    B --> C[Load Data]
    C --> D{Training Mode?}

    D -->|Yes| E[Training Loop]
    E --> F[Play Episode]
    F --> G[Get State]
    G --> H[Agent Takes Action]
    H --> I[Environment Step]
    I --> J[Calculate Reward]
    J --> K[Train Agent]
    K --> L{Episode Done?}
    L -->|No| G
    L -->|Yes| M{More Episodes?}
    M -->|Yes| F
    M -->|No| N[Save Model]

    D -->|No| O[Test/Real Mode]
    O --> P[Load Trained Model]
    P --> Q[Get Market State]
    Q --> R[Agent Predicts Action]
    R --> S[Execute Trade]
    S --> T{Continue Trading?}
    T -->|Yes| Q
    T -->|No| U[End]

```

Here are the main components:

1. Environment (`MultiStockEnv` class):

```mermaid
graph TD
    A[MultiStockEnv] --> B[Initialize Parameters]
    B --> C[State Space]
    B --> D[Action Space]

    E[reset] --> F[Reset Portfolio]
    F --> G[Reset Trading Cycle]

    H[step] --> I[Execute Action]
    I --> J[Calculate Reward]
    J --> K[Update State]
    K --> L[Return Results]

```

1. Trading Logic:

```mermaid
graph TD
    A[Trading Cycle] --> B{Action Type}
    B -->|0| C[No Trade]
    B -->|1| D[Start Buy]
    B -->|2| E[Start Sell]

    D --> F[Monitor Buy Positions]
    E --> G[Monitor Sell Positions]

    F --> H{Check Conditions}
    H -->|TP Hit| I[Take Profit]
    H -->|SL Hit| J[Stop Loss]
    H -->|Neither| K[Continue]

    G --> L{Check Conditions}
    L -->|TP Hit| M[Take Profit]
    L -->|SL Hit| N[Stop Loss]
    L -->|Neither| O[Continue]

```

1. Agent Logic (`DQNAgent` class):

```mermaid
graph TD
    A[DQNAgent] --> B[Initialize Model]
    B --> C[Linear Model]

    D[act] --> E{Random Action?}
    E -->|Yes| F[Random Choice]
    E -->|No| G[Model Prediction]

    H[train] --> I[Calculate Target]
    I --> J[Update Weights]
    J --> K[Update Epsilon]

```

Key components explanation:

1. **Data Processing**:
- Uses historical price data with OHLC and moving averages
- Scales data using StandardScaler
- Splits data into training and testing sets
1. **Environment**:
- Manages trading state and actions
- Handles position entry/exit
- Calculates rewards based on profit/loss
- Implements grid trading strategy with multiple levels
1. **Agent**:
- Uses linear model for Q-learning
- Implements epsilon-greedy exploration
- Handles model training and prediction
1. **Trading Strategy**:
- Uses 3-level grid trading system
- Implements take profit and stop loss
- Manages position sizing and risk
1. **Modes of Operation**:
- Training: Learns from historical data
- Testing: Evaluates strategy on unseen data
- Real: Live trading with database integration