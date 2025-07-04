"""
Chess AI API Backend
RESTful API for playing chess against your AI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import copy
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import threading
import json

from chess_engine import ChessBoard, Color, PieceType, Piece, Move
from chess_ai_integration import ChessAI
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# In-memory game storage (use Redis/database in production)
active_games: Dict[str, Dict] = {}
game_cleanup_lock = threading.Lock()

class GameSession:
    """Manages a single chess game session"""
    
    def __init__(self, game_id: str, player_color: str = "white"):
        self.game_id = game_id
        self.board = ChessBoard()
        self.ai = ChessAI()
        self.player_color = Color.WHITE if player_color.lower() == "white" else Color.BLACK
        self.ai_color = Color.BLACK if self.player_color == Color.WHITE else Color.WHITE
        self.move_history = []
        self.game_status = "active"  # active, check, checkmate, stalemate, draw
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert game state to dictionary for JSON response"""
        return {
            "game_id": self.game_id,
            "board": self.get_board_state(),
            "current_player": self.board.current_player.value,
            "player_color": self.player_color.value,
            "ai_color": self.ai_color.value,
            "move_history": [str(move) for move in self.move_history],
            "game_status": self.game_status,
            "in_check": self.board.is_in_check(self.board.current_player),
            "legal_moves": [str(move) for move in self.board.get_all_legal_moves()],
            "position_evaluation": self.ai.evaluate_position(self.board) if self.ai.model else 0.0,
            "last_activity": self.last_activity.isoformat()
        }
    
    def get_board_state(self) -> List[List[Optional[str]]]:
        """Get current board state as 2D array"""
        board_state = []
        for row in range(8):
            board_row = []
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece:
                    # Return piece notation (e.g., "wK" for white king, "bP" for black pawn)
                    color_char = "w" if piece.color == Color.WHITE else "b"
                    piece_char = {
                        PieceType.PAWN: "P", PieceType.ROOK: "R", PieceType.KNIGHT: "N",
                        PieceType.BISHOP: "B", PieceType.QUEEN: "Q", PieceType.KING: "K"
                    }[piece.piece_type]
                    board_row.append(color_char + piece_char)
                else:
                    board_row.append(None)
            board_state.append(board_row)
        return board_state
    
    def make_player_move(self, move_str: str) -> Dict:
        """Process player move"""
        self.last_activity = datetime.now()
        
        if self.board.current_player != self.player_color:
            return {"success": False, "error": "Not your turn"}
        
        if self.game_status != "active":
            return {"success": False, "error": f"Game is {self.game_status}"}
        
        # Parse and validate move
        move = self.board.parse_move(move_str)
        if not move or not self.board.is_legal_move(move):
            return {"success": False, "error": "Invalid move"}
        
        # Make the move
        success = self.board.make_move(move)
        if not success:
            return {"success": False, "error": "Failed to make move"}
        
        self.move_history.append(move)
        self.update_game_status()
        
        return {"success": True, "move": str(move)}
    
    def make_ai_move(self) -> Dict:
        """Process AI move"""
        self.last_activity = datetime.now()
        
        if self.board.current_player != self.ai_color:
            return {"success": False, "error": "Not AI's turn"}
        
        if self.game_status != "active":
            return {"success": False, "error": f"Game is {self.game_status}"}
        
        # Get AI move
        start_time = time.time()
        ai_move = self.ai.select_best_move(self.board)
        think_time = time.time() - start_time
        
        if not ai_move:
            return {"success": False, "error": "AI couldn't find a move"}
        
        # Make the move
        success = self.board.make_move(ai_move)
        if not success:
            return {"success": False, "error": "AI made invalid move"}
        
        self.move_history.append(ai_move)
        self.update_game_status()
        
        return {
            "success": True, 
            "move": str(ai_move),
            "think_time": round(think_time, 2)
        }
    
    def update_game_status(self):
        """Update game status based on current position"""
        current_player = self.board.current_player
        
        if self.board.is_checkmate(current_player):
            winner = "black" if current_player == Color.WHITE else "white"
            self.game_status = f"checkmate_{winner}_wins"
        elif self.board.is_stalemate(current_player):
            self.game_status = "stalemate"
        elif self.board.is_in_check(current_player):
            self.game_status = "check"
        else:
            self.game_status = "active"

# API Routes

@app.route('/')
def home():
    """Serve the chess frontend HTML file"""
    try:
        frontend_path = Path(__file__).parent / 'chess_frontend.html'
        if frontend_path.exists():
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <h1>Frontend Not Found</h1>
            <p>Please make sure 'chess_frontend.html' is in the same directory as the backend script.</p>
            <p>You can also access the API directly:</p>
            <ul>
                <li>POST /api/games - Create new game</li>
                <li>GET /api/games/{id} - Get game state</li>
                <li>POST /api/games/{id}/moves - Make a move</li>
            </ul>
            """, 404
    except Exception as e:
        return f"Error loading frontend: {e}", 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_games": len(active_games)
    })

@app.route('/api/games', methods=['POST'])
def create_game():
    """Create a new chess game"""
    try:
        data = request.get_json() or {}
        player_color = data.get('player_color', 'white').lower()
        
        if player_color not in ['white', 'black']:
            return jsonify({"error": "player_color must be 'white' or 'black'"}), 400
        
        # Generate unique game ID
        game_id = str(uuid.uuid4())
        
        # Create new game session
        game = GameSession(game_id, player_color)
        active_games[game_id] = game
        
        # If player is black, make AI move first
        response_data = game.to_dict()
        if player_color == 'black':
            ai_result = game.make_ai_move()
            response_data.update(ai_result)
            response_data.update(game.to_dict())
        
        return jsonify(response_data), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games/<game_id>', methods=['GET'])
def get_game(game_id: str):
    """Get current game state"""
    try:
        game = active_games.get(game_id)
        if not game:
            return jsonify({"error": "Game not found"}), 404
        
        return jsonify(game.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games/<game_id>/moves', methods=['POST'])
def make_move(game_id: str):
    """Make a move in the game"""
    try:
        game = active_games.get(game_id)
        if not game:
            return jsonify({"error": "Game not found"}), 404
        
        data = request.get_json()
        if not data or 'move' not in data:
            return jsonify({"error": "Move required"}), 400
        
        move_str = data['move']
        
        # Make player move
        player_result = game.make_player_move(move_str)
        if not player_result['success']:
            return jsonify(player_result), 400
        
        # Prepare response
        response_data = {
            "player_move": player_result,
            "game_state": game.to_dict()
        }
        
        # If game is still active and it's AI's turn, make AI move
        if (game.game_status == "active" and 
            game.board.current_player == game.ai_color):
            
            ai_result = game.make_ai_move()
            response_data["ai_move"] = ai_result
            response_data["game_state"] = game.to_dict()
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games/<game_id>/ai-move', methods=['POST'])
def request_ai_move(game_id: str):
    """Request AI to make a move (if it's AI's turn)"""
    try:
        game = active_games.get(game_id)
        if not game:
            return jsonify({"error": "Game not found"}), 404
        
        ai_result = game.make_ai_move()
        response_data = {
            "ai_move": ai_result,
            "game_state": game.to_dict()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games/<game_id>', methods=['DELETE'])
def delete_game(game_id: str):
    """Delete a game"""
    try:
        if game_id in active_games:
            del active_games[game_id]
            return jsonify({"message": "Game deleted"})
        else:
            return jsonify({"error": "Game not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games', methods=['GET'])
def list_games():
    """List all active games (for debugging)"""
    try:
        games_info = []
        for game_id, game in active_games.items():
            games_info.append({
                "game_id": game_id,
                "player_color": game.player_color.value,
                "game_status": game.game_status,
                "moves_count": len(game.move_history),
                "created_at": game.created_at.isoformat(),
                "last_activity": game.last_activity.isoformat()
            })
        
        return jsonify({
            "active_games": len(active_games),
            "games": games_info
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Background cleanup task
def cleanup_old_games():
    """Remove games older than 2 hours"""
    try:
        with game_cleanup_lock:
            current_time = datetime.now()
            expired_games = []
            
            for game_id, game in active_games.items():
                if current_time - game.last_activity > timedelta(hours=2):
                    expired_games.append(game_id)
            
            for game_id in expired_games:
                del active_games[game_id]
                print(f"Cleaned up expired game: {game_id}")
                
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Schedule cleanup every 30 minutes
def schedule_cleanup():
    cleanup_old_games()
    threading.Timer(1800, schedule_cleanup).start()  # 30 minutes

# Basic HTML template for testing
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chess AI API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .board { display: grid; grid-template-columns: repeat(8, 50px); gap: 1px; margin: 20px 0; }
        .square { width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
        .white { background-color: #f0d9b5; }
        .black { background-color: #b58863; }
        .game-info { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; }
        input { padding: 8px; margin: 5px; font-size: 16px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>🏛️ Chess AI API</h1>
    <p>RESTful API for playing chess against your trained AI</p>
    
    <div class="game-info">
        <h3>API Endpoints:</h3>
        <ul>
            <li><strong>POST /api/games</strong> - Create new game</li>
            <li><strong>GET /api/games/{game_id}</strong> - Get game state</li>
            <li><strong>POST /api/games/{game_id}/moves</strong> - Make a move</li>
            <li><strong>DELETE /api/games/{game_id}</strong> - Delete game</li>
        </ul>
    </div>
    
    <div class="controls">
        <h3>Quick Test:</h3>
        <button onclick="createGame('white')">New Game (Play as White)</button>
        <button onclick="createGame('black')">New Game (Play as Black)</button>
        <br>
        <input type="text" id="moveInput" placeholder="Enter move (e.g., e2e4)" style="width: 200px;">
        <button onclick="makeMove()">Make Move</button>
        <button onclick="getGameState()">Refresh Game</button>
    </div>
    
    <div id="status"></div>
    <div id="gameInfo"></div>
    
    <script>
        let currentGameId = null;
        
        async function createGame(playerColor) {
            try {
                const response = await fetch('/api/games', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({player_color: playerColor})
                });
                
                const data = await response.json();
                if (response.ok) {
                    currentGameId = data.game_id;
                    showStatus(`Game created! ID: ${currentGameId}`, 'success');
                    displayGameState(data);
                } else {
                    showStatus(`Error: ${data.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            }
        }
        
        async function makeMove() {
            if (!currentGameId) {
                showStatus('No active game. Create a game first.', 'error');
                return;
            }
            
            const move = document.getElementById('moveInput').value.trim();
            if (!move) {
                showStatus('Please enter a move', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/games/${currentGameId}/moves`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({move: move})
                });
                
                const data = await response.json();
                if (response.ok) {
                    showStatus(`Move made: ${move}`, 'success');
                    displayGameState(data.game_state);
                    document.getElementById('moveInput').value = '';
                } else {
                    showStatus(`Error: ${data.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            }
        }
        
        async function getGameState() {
            if (!currentGameId) {
                showStatus('No active game', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/games/${currentGameId}`);
                const data = await response.json();
                if (response.ok) {
                    displayGameState(data);
                } else {
                    showStatus(`Error: ${data.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            }
        }
        
        function displayGameState(gameState) {
            const gameInfo = document.getElementById('gameInfo');
            gameInfo.innerHTML = `
                <div class="game-info">
                    <h3>Game State</h3>
                    <p><strong>Status:</strong> ${gameState.game_status}</p>
                    <p><strong>Current Turn:</strong> ${gameState.current_player}</p>
                    <p><strong>You are:</strong> ${gameState.player_color}</p>
                    <p><strong>In Check:</strong> ${gameState.in_check ? 'Yes' : 'No'}</p>
                    <p><strong>Position Eval:</strong> ${gameState.position_evaluation.toFixed(2)}</p>
                    <p><strong>Moves:</strong> ${gameState.move_history.join(', ')}</p>
                    <p><strong>Legal Moves:</strong> ${gameState.legal_moves.slice(0, 10).join(', ')}${gameState.legal_moves.length > 10 ? '...' : ''}</p>
                </div>
            `;
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.className = `status ${type}`;
            status.textContent = message;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🏛️ Starting Chess AI API Server...")
    print("📡 API will be available at: http://localhost:5000")
    print("🌐 Web interface at: http://localhost:5000")
    print("📚 API documentation at endpoints listed above")
    
    # Start cleanup scheduler
    schedule_cleanup()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)