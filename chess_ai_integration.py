"""
Chess AI Integration & Game Interface
Connects the trained neural network with your chess engine
"""

import torch
import numpy as np
import random
import copy

from typing import Optional, Tuple, List
from chess_engine import ChessBoard, Color, PieceType, Piece, Move
from chess_ai_model import ChessNet, ChessTrainer
from chess_engine_bridge import ChessEngineBridge
import time

class ChessAI:
    """AI player that uses the trained neural network"""
    
    def __init__(self, model_path: str = 'best_chess_model.pth', device: str = 'cpu'):
        self.device = device
        self.bridge = ChessEngineBridge()
        
        # Load the trained model
        self.model = ChessNet()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"✓ Loaded trained AI model from {model_path}")
        except FileNotFoundError:
            print(f"⚠ Model file {model_path} not found. AI will make random moves.")
            self.model = None
        
        self.trainer = ChessTrainer(self.model, device) if self.model else None
    
    def board_to_features(self, board: ChessBoard) -> Tuple[np.ndarray, np.ndarray]:
        """Convert chess board to neural network input features"""
        # Convert to python-chess format for feature extraction
        chess_board = self.bridge.your_board_to_python_chess(board)
        
        # Use the same feature extraction as training
        from chess_data_pipe import ChessDataProcessor
        processor = ChessDataProcessor()
        
        position_features = processor.board_to_features(chess_board)
        additional_features = processor.get_additional_features(chess_board)
        
        return position_features, additional_features
    
    def get_legal_moves_with_squares(self, board: ChessBoard) -> List[Tuple[Move, int, int]]:
        """Get legal moves with their from/to square indices"""
        legal_moves = board.get_all_legal_moves()
        moves_with_squares = []
        
        for move in legal_moves:
            from_row, from_col = move.from_pos
            to_row, to_col = move.to_pos
            
            # Convert to square indices (0-63)
            from_square = from_row * 8 + from_col
            to_square = to_row * 8 + to_col
            
            moves_with_squares.append((move, from_square, to_square))
        
        return moves_with_squares
    
    def evaluate_move_heuristically(self, board: ChessBoard, move: Move) -> float:
        """Add chess knowledge to improve move selection"""
        score = 0.0
        # Piece values for tactical evaluation

        piece_values = {
            PieceType.PAWN: 1, PieceType.KNIGHT: 3, PieceType.BISHOP: 3,
            PieceType.ROOK: 5, PieceType.QUEEN: 9, PieceType.KING: 0
        }
        
        # Make temp moves to eval position
        temp_board = copy.deepcopy(board)
        temp_board.make_move(move, check_legality=False)
        
        # Highest priority is checkmating
        opponent_color = Color.BLACK if move.piece.color == Color.WHITE else Color.WHITE
        if temp_board.is_checkmate(opponent_color):
            return 1000 # HUGEEEEE BONUSS HAHAHAHAHAH
        
        # If move puts in check
        if temp_board.is_in_check(opponent_color):
            score += 15 # good bonus
        
        # 1. Prioritize captures (especially good trades)
        if move.captured_piece:
            captured_value = piece_values.get(move.captured_piece.piece_type, 0)
            moving_value = piece_values.get(move.piece.piece_type, 0)
            score += captured_value * 10  # Big bonus for captures
            # Avoid bad trades (don't trade queen for pawn)

            if captured_value < moving_value:
                score -= (moving_value - captured_value) * 5

        to_row, to_col = move.to_pos
        
        # Check if the moved piece can be captured
        if temp_board.is_square_attacked(to_row, to_col, opponent_color):

            piece_value = piece_values.get(move.piece.piece_type, 0)
            score -= piece_value * 8  # Heavy penalty for hanging pieces

        # BETETER ENDGAME YEAHHHHH
        total_material = self.count_total_material(board)
        if total_material <= 20:
            score += self.evaluate_endgame_factors(temp_board, move)
        
        
        if len(board.move_history) < 20:
            to_row, to_col = move.to_pos
            if 2 <= to_row <= 5 and 2 <= to_col <= 5:  # Center squares
                score += 2

        # 4. Castling bonus
        if move.is_castling:
            score += 5
            
        # piece development
        if len(board.move_history) < 16:
            if self.piece_already_moved(board, move.piece, move.from_pos):
                score -= 3

        # 5. Avoid moving into check
        if temp_board.is_in_check(move.piece.color):
            score -= 20
        return score
    
    def count_total_material(self, board: ChessBoard) -> int:
        """Count total material on board"""
        piece_values = {
            PieceType.PAWN: 1, PieceType.KNIGHT: 3, PieceType.BISHOP: 3,
            PieceType.ROOK: 5, PieceType.QUEEN: 9, PieceType.KING: 0
        }
        
        total = 0
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    total += piece_values.get(piece.piece_type, 0)
        return total
    
    def evaluate_endgame_factors(self, board: ChessBoard, move: Move) -> float:
        """Special evaluation for endgame positions"""
        score = 0.0
        
        # In endgame, centralize king
        if move.piece.piece_type == PieceType.KING:
            to_row, to_col = move.to_pos
            # Bonus for moving king to center
            center_distance = abs(3.5 - to_row) + abs(3.5 - to_col)
            score += (7 - center_distance) * 2
        
        # Promote pawns aggressively
        if move.piece.piece_type == PieceType.PAWN:
            to_row, to_col = move.to_pos
            if move.piece.color == Color.WHITE:
                score += (6 - to_row) * 3  # Closer to promotion
            else:
                score += (to_row - 1) * 3  # Closer to promotion
        
        # Use heavy pieces to support mate
        if move.piece.piece_type in [PieceType.QUEEN, PieceType.ROOK]:
            score += 3
        
        return score
    
    def piece_already_moved(self, board: ChessBoard, piece: Piece, from_pos: Tuple[int, int]) -> bool:
        """Check if this piece has already moved in the opening"""
        piece_first_moves = {
            # Track if pieces have moved from starting positions
            PieceType.KNIGHT: [(0, 1), (0, 6), (7, 1), (7, 6)],
            PieceType.BISHOP: [(0, 2), (0, 5), (7, 2), (7, 5)],
            PieceType.QUEEN: [(0, 3), (7, 3)],
        }
        
        starting_positions = piece_first_moves.get(piece.piece_type, [])
        
        # Count how many times this piece type has moved
        move_count = 0
        for past_move in board.move_history:
            if past_move.piece.piece_type == piece.piece_type and past_move.piece.color == piece.color:
                move_count += 1
        
        # Penalize moving same piece multiple times in opening
        return move_count > 0 and from_pos not in starting_positions
    
    def select_best_move(self, board: ChessBoard) -> Optional[Move]:
        """Select the best move using AI + chess heuristics"""
        if not self.model:
            return self.select_random_move(board)
        
        try:
            # Get position features
            position_features, additional_features = self.board_to_features(board)
            
            # Get AI prediction
            predicted_from, predicted_to = self.trainer.predict_move(
                position_features, additional_features
            )
            
            # Get all legal moves
            legal_moves_with_squares = self.get_legal_moves_with_squares(board)
            
            if not legal_moves_with_squares:
                return None
            
            # Check for material advantage - if big advantage, prioritize checkmate
            material_advantage = self.calculate_material_advantage(board)
            is_winning = material_advantage > 5  # Significant material advantage
            
            # Combine AI prediction with chess heuristics
            best_move = None
            best_total_score = float('-inf')
            
            for move, from_square, to_square in legal_moves_with_squares:
                # AI prediction score
                ai_score = 0
                if from_square == predicted_from and to_square == predicted_to:
                    ai_score = 10  # Exact match bonus
                elif from_square == predicted_from:
                    ai_score = 3   # Partial match
                elif to_square == predicted_to:
                    ai_score = 2   # Partial match
                
                # Chess heuristics score
                heuristic_score = self.evaluate_move_heuristically(board, move)
                
                # If winning, prioritize heuristics more heavily (especially checkmate detection)
                if is_winning:
                    total_score = heuristic_score + (ai_score * 0.2)  # Heavy heuristic weight
                else:
                    total_score = heuristic_score + (ai_score * 0.5)  # Balanced
                
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_move = move
            
            if best_move:
                # Special logging for checkmate moves
                if best_total_score >= 1000:
                    print(f"🏆 AI found CHECKMATE: {best_move}!")
                elif best_total_score >= 15:
                    print(f"⚔️ AI giving CHECK: {best_move} (score: {best_total_score:.1f})")
                else:
                    print(f"🧠 AI selected: {best_move} (score: {best_total_score:.1f})")
                return best_move
            
            # Fallback to first legal move
            return legal_moves_with_squares[0][0]
            
        except Exception as e:
            print(f"⚠ AI prediction failed: {e}. Using random move.")
            return self.select_random_move(board)
        
    def calculate_material_advantage(self, board: ChessBoard) -> float:
        """Calculate material advantage for current AI color"""
        piece_values = {
            PieceType.PAWN: 1, PieceType.KNIGHT: 3, PieceType.BISHOP: 3,
            PieceType.ROOK: 5, PieceType.QUEEN: 9, PieceType.KING: 0
        }
        
        ai_material = 0
        opponent_material = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    value = piece_values.get(piece.piece_type, 0)
                    if piece.color == board.current_player:  # AI's color
                        ai_material += value
                    else:
                        opponent_material += value
        
        return ai_material - opponent_material
    
    def select_random_move(self, board: ChessBoard) -> Optional[Move]:
        """Fallback: select a random legal move"""
        legal_moves = board.get_all_legal_moves()
        if legal_moves:
            move = random.choice(legal_moves)
            print(f"🎲 Random move: {move}")
            return move
        return None
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """Evaluate the current position"""
        if not self.model:
            return 0.0
        
        try:
            position_features, additional_features = self.board_to_features(board)
            evaluation = self.trainer.evaluate_position(position_features, additional_features)
            return evaluation
        except Exception as e:
            print(f"⚠ Position evaluation failed: {e}")
            return 0.0

class ChessGame:
    """Main game controller for human vs AI"""
    
    def __init__(self, ai_color: Color = Color.BLACK):
        self.board = ChessBoard()
        self.ai = ChessAI()
        self.ai_color = ai_color
        self.game_history = []
        
        print("=" * 60)
        print("🏛️  CHESS AI GAME")
        print("=" * 60)
        print(f"You are playing as: {Color.WHITE.value.upper() if ai_color == Color.BLACK else Color.BLACK.value.upper()}")
        print(f"AI is playing as: {ai_color.value.upper()}")
        print("Enter moves in format: e2e4 (from square to square)")
        print("Type 'quit' to exit, 'eval' to see position evaluation")
        print("=" * 60)
    
    def display_game_state(self):
        """Display the current game state"""
        self.board.display()
        
        # Show position evaluation
        if self.ai.model:
            eval_score = self.ai.evaluate_position(self.board)
            if eval_score > 0.1:
                eval_str = f"🟢 +{eval_score:.2f} (White advantage)"
            elif eval_score < -0.1:
                eval_str = f"🔴 {eval_score:.2f} (Black advantage)"
            else:
                eval_str = f"⚖️  {eval_score:.2f} (Equal)"
            print(f"Position evaluation: {eval_str}")
        print()
    
    def get_human_move(self) -> Optional[Move]:
        """Get move input from human player"""
        while True:
            try:
                move_input = input(f"Your move ({self.board.current_player.value}): ").strip().lower()
                
                if move_input == 'quit':
                    return None
                elif move_input == 'eval':
                    eval_score = self.ai.evaluate_position(self.board)
                    print(f"Position evaluation: {eval_score:.3f}")
                    continue
                elif move_input == 'help':
                    self.show_legal_moves()
                    continue
                
                # Parse the move
                move = self.board.parse_move(move_input)
                if move and self.board.is_legal_move(move):
                    return move
                else:
                    print("❌ Invalid move! Try again (e.g., e2e4)")
                    
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                return None
            except Exception as e:
                print(f"❌ Error: {e}. Try again.")
    
    def show_legal_moves(self):
        """Show all legal moves for the current player"""
        legal_moves = self.board.get_all_legal_moves()
        if legal_moves:
            move_strs = [str(move) for move in legal_moves[:10]]  # Show first 10
            print(f"Legal moves: {', '.join(move_strs)}")
            if len(legal_moves) > 10:
                print(f"... and {len(legal_moves) - 10} more")
        else:
            print("No legal moves available!")
    
    def make_ai_move(self) -> bool:
        """Make AI move"""
        print(f"🤖 AI ({self.ai_color.value}) is thinking...")
        
        start_time = time.time()
        ai_move = self.ai.select_best_move(self.board)
        think_time = time.time() - start_time
        
        if ai_move:
            success = self.board.make_move(ai_move)
            if success:
                print(f"✓ AI played: {ai_move} (thought for {think_time:.1f}s)")
                self.game_history.append(ai_move)
                return True
            else:
                print("❌ AI attempted illegal move!")
                return False
        else:
            print("❌ AI couldn't find a move!")
            return False
    
    def check_game_end(self) -> Optional[str]:
        """Check if game has ended"""
        current_player = self.board.current_player
        
        if self.board.is_checkmate(current_player):
            winner = "Black" if current_player == Color.WHITE else "White"
            return f"Checkmate! {winner} wins! 🎉"
        elif self.board.is_stalemate(current_player):
            return "Stalemate! It's a draw! 🤝"
        elif self.board.is_in_check(current_player):
            print(f"⚠️  {current_player.value.capitalize()} is in check!")
        
        return None
    
    def play(self):
        """Main game loop"""
        while True:
            self.display_game_state()
            
            # Check for game end
            end_result = self.check_game_end()
            if end_result:
                print(end_result)
                break
            
            # Current player's turn
            if self.board.current_player == self.ai_color:
                # AI turn
                if not self.make_ai_move():
                    print("AI failed to move. Game ended.")
                    break
            else:
                # Human turn
                human_move = self.get_human_move()
                if human_move is None:
                    print("Game ended by player.")
                    break
                
                success = self.board.make_move(human_move)
                if success:
                    print(f"✓ You played: {human_move}")
                    self.game_history.append(human_move)
                else:
                    print("❌ Invalid move!")
                    continue
            
            print("-" * 60)
        
        # Game summary
        print(f"\nGame finished after {len(self.game_history)} moves.")
        if len(self.game_history) > 0:
            print("Move history:", [str(move) for move in self.game_history[-10:]])

def main():
    """Main entry point"""
    print("🏛️  Welcome to Chess AI!")
    print("\nChoose game mode:")
    print("1. Play as White (AI plays Black)")
    print("2. Play as Black (AI plays White)")
    print("3. Watch AI vs AI")
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == '1':
                game = ChessGame(ai_color=Color.BLACK)
                break
            elif choice == '2':
                game = ChessGame(ai_color=Color.WHITE)
                break
            elif choice == '3':
                print("AI vs AI mode not implemented yet!")
                continue
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
    
    # Start the game
    try:
        game.play()
    except KeyboardInterrupt:
        print("\nGame interrupted. Goodbye!")
    except Exception as e:
        print(f"Game error: {e}")

if __name__ == "__main__":
    main()