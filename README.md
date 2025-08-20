AI Chess Engine

A complete chess AI system built from scratch, featuring a custom chess engine, deep learning neural network, and modern web interface. This project demonstrates full-stack development, machine learning, and game development skills.


Features

Core Chess Engine
- Complete rule implementation - All chess rules including castling, en passant, pawn promotion
- Move validation - Ensures only legal moves are allowed
- Game state detection - Checkmate, stalemate, and draw detection
- Position evaluation - Material and positional assessment

AI Neural Network
- Deep learning model - Convolutional neural network for position evaluation
- Move prediction - Trained on thousands of chess games
- Hybrid decision making - Combines neural network with chess heuristics
- Endgame expertise - Special logic for checkmate scenarios

Web Interface
- Real-time gameplay - Instant move updates and game state
- Interactive board - Click-to-move with visual feedback
- Position analysis - Live evaluation bar showing game balance


```
├── Core Engine (Python)
│   ├── chess_engine.py          # Chess rules and game logic
│   ├── chess_engine_bridge.py   # Interface adapter
│   └── chess_ai_model.py        # Neural network architecture
│
├── Data Pipeline
│   ├── csv_chess_data_pipe.py   # Training data processing
│   └── chess_data/              # Processed datasets
│
├── AI Integration
│   ├── chess_ai_integration.py  # AI decision engine
│   └── best_chess_model.pth     # Trained neural network
│
├── Web Application
│   ├── chess_api_backend.py     # Flask REST API
│   └── chess_frontend.html      # Interactive web interface
│
└── Documentation
    ├── README.md                 # This file
    └── requirements.txt          # Python dependencies
```

The chess AI uses a multi-layered approach:

1. Convolutional Neural Network
   - 8×8×12 board representation
   - 5 convolutional layers for pattern recognition
   - Dual-head output: move prediction + position evaluation

2. Chess Knowledge Integration
   - Material balance evaluation
   - Tactical pattern recognition
   - Endgame-specific heuristics
   - Checkmate detection algorithms

3. Hybrid Decision Engine
   - Neural network provides strategic guidance
   - Rule-based system prevents blunders
   - Adaptive weighting based on game phase

AI Capabilities:
- Plays at intermediate level (~1400-1600 ELO estimated)
- Recognizes tactical patterns and delivers checkmates
- Adapts strategy based on game phase

Project Structure

**`chess_engine.py`** - Complete chess implementation
- Object-oriented design with `ChessBoard`, `Piece`, and `Move` classes
- Full rule compliance including special moves
- Efficient move generation and validation

**`chess_ai_model.py`** - Deep learning architecture
- PyTorch-based convolutional neural network
- Multi-task learning: move prediction + position evaluation
- Training pipeline with validation and early stopping

**`chess_api_backend.py`** - Web API server
- RESTful endpoints for game management
- Real-time game state updates
- CORS-enabled for frontend integration

**`chess_frontend.html`** - Modern web interface
- Interactive chess board with drag-and-drop
- Real-time game analysis and move history
