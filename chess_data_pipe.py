"""
Chess Data Pipeline for AI Training
Handles PGN parsing, position extraction, and dataset creation
"""

import chess
import chess.pgn
import numpy as np
import pandas as pd
import pickle
import gzip
import io
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import requests
from tqdm import tqdm

class ChessDataProcessor:
    def __init__(self, output_dir: str = "chess_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Piece values for position evaluation
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King safety handled separately
        }
    
    def load_csv_games(self, csv_file: str) -> pd.DataFrame:
        """Load chess games from CSV file"""
        print(f"Loading games from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} games from CSV")
            print(f"Columns: {list(df.columns)}")
            
            # Display basic statistics
            print(f"\nDataset Statistics:")
            print(f"Total games: {len(df)}")
            print(f"Rated games: {df['rated'].sum() if 'rated' in df.columns else 'N/A'}")
            if 'white_rating' in df.columns and 'black_rating' in df.columns:
                print(f"Average White rating: {df['white_rating'].mean():.0f}")
                print(f"Average Black rating: {df['black_rating'].mean():.0f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()
        
    def parse_moves_string(self, moves_str: str) -> List[str]:
        """Parse moves string into list of individual moves"""
        if pd.isna(moves_str) or not moves_str:
            return []
        
        # Remove move numbers and split by spaces
        moves = moves_str.replace('\n', ' ').split()
        
        # Filter out move numbers (like "1.", "2.", etc.)
        parsed_moves = []
        for move in moves:
            if not move.endswith('.') and move not in ['1-0', '0-1', '1/2-1/2', '*']:
                parsed_moves.append(move)
        
        return parsed_moves
    
    def board_to_features(self, board: chess.Board) -> np.ndarray:
        """Convert chess position to feature vector"""
        # 8x8x12 representation (6 piece types x 2 colors)
        features = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_to_channel = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }
        
        # Fill piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Convert to our coordinate system
                col = square % 8
                channel = piece_to_channel[(piece.piece_type, piece.color)]
                features[row, col, channel] = 1.0
        
        return features
    
    def get_additional_features(self, board: chess.Board) -> np.ndarray:
        """Extract additional game state features"""
        features = []
        
        # Turn (1 for white, 0 for black)
        features.append(1.0 if board.turn == chess.WHITE else 0.0)
        
        # Castling rights
        features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
        features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
        features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
        features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        
        # En passant
        features.append(1.0 if board.ep_square is not None else 0.0)
        
        # Check status
        features.append(1.0 if board.is_check() else 0.0)
        
        # Material count difference
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                           for piece_type, value in self.piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                           for piece_type, value in self.piece_values.items())
        features.append((white_material - black_material) / 39.0)  # Normalize by max material
        
        return np.array(features, dtype=np.float32)
    
    def get_game_result(self, row: pd.Series) -> str:
        """Extract game result from CSV row"""
        # Try different possible column names for result
        if 'winner' in row and pd.notna(row['winner']):
            if row['winner'] == 'white':
                return "1-0"
            elif row['winner'] == 'black':
                return "0-1"
            else:
                return "1/2-1/2"  # Draw
        elif 'result' in row:
            return str(row['result'])
        elif 'victory_status' in row and pd.notna(row['victory_status']):
            # Assume draws if no clear winner
            return "1/2-1/2"
        else:
            return "1/2-1/2"  # Default to draw
    
    def evaluate_position(self, board: chess.Board, game_result: str) -> float:
        """Simple position evaluation based on game outcome"""
        if game_result == "1-0":  # White wins
            return 1.0 if board.turn == chess.WHITE else -1.0
        elif game_result == "0-1":  # Black wins
            return -1.0 if board.turn == chess.WHITE else 1.0
        else:  # Draw
            return 0.0
    
    def process_csv_games(self, csv_file: str, max_games: int = 10000, 
                         min_rating: int = 1200) -> List[Dict]:
        """Process CSV file and extract training data"""
        
        # Load CSV data
        df = self.load_csv_games(csv_file)
        if df.empty:
            print("No data loaded from CSV file!")
            return []
        
        # Filter games by rating if available
        if 'white_rating' in df.columns and 'black_rating' in df.columns:
            df = df[(df['white_rating'] >= min_rating) & (df['black_rating'] >= min_rating)]
            print(f"After rating filter (min {min_rating}): {len(df)} games")
        
        # Filter rated games if available
        if 'rated' in df.columns:
            df = df[df['rated'] == True]
            print(f"After rated filter: {len(df)} games")
        
        # Limit number of games
        if len(df) > max_games:
            df = df.sample(n=max_games, random_state=42)
            print(f"Randomly sampled {max_games} games")
        
        training_data = []
        games_processed = 0
        
        print(f"Processing {len(df)} games...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing games"):
            try:
                # Parse moves
                moves_str = row['moves'] if 'moves' in row else ''
                moves = self.parse_moves_string(moves_str)
                
                if len(moves) < 10:  # Skip very short games
                    continue
                
                # Get game result
                result = self.get_game_result(row)
                
                # Process the game
                board = chess.Board()
                
                # Extract positions from the game
                for i, move_str in enumerate(moves):
                    try:
                        # Skip opening moves (first 6 moves)
                        if i < 6:
                            move = chess.Move.from_uci(move_str) if len(move_str) == 4 else board.parse_san(move_str)
                            if move in board.legal_moves:
                                board.push(move)
                            else:
                                break  # Invalid move, skip this game
                            continue
                        
                        # Skip endgame (last 10 moves) to focus on middlegame
                        if i >= len(moves) - 10:
                            break
                        
                        # Create training sample
                        position_features = self.board_to_features(board)
                        additional_features = self.get_additional_features(board)
                        
                        # Parse the move
                        try:
                            if len(move_str) == 4:  # UCI format (e2e4)
                                move = chess.Move.from_uci(move_str)
                            else:  # SAN format (Nf3)
                                move = board.parse_san(move_str)
                        except:
                            break  # Invalid move, skip rest of game
                        
                        if move not in board.legal_moves:
                            break  # Invalid move, skip rest of game
                        
                        # Get move squares
                        from_square = move.from_square
                        to_square = move.to_square
                        
                        # Simple position evaluation
                        position_eval = self.evaluate_position(board, result)
                        
                        training_sample = {
                            'position': position_features,
                            'additional_features': additional_features,
                            'best_move_from': from_square,
                            'best_move_to': to_square,
                            'position_eval': position_eval,
                            'game_phase': min(i / len(moves), 1.0),  # Game phase (0=opening, 1=endgame)
                        }
                        
                        training_data.append(training_sample)
                        board.push(move)
                        
                    except Exception as e:
                        # Skip invalid moves
                        continue
                
                games_processed += 1
                if games_processed % 1000 == 0:
                    print(f"Processed {games_processed} games, {len(training_data)} positions")
                    
            except Exception as e:
                # Skip problematic games
                continue
        
        print(f"Finished processing. Total positions: {len(training_data)}")
        return training_data
    
    def create_dataset(self, training_data: List[Dict], 
                      validation_split: float = 0.2) -> Tuple[Dict, Dict]:
        """Create training and validation datasets"""
        # Shuffle data
        np.random.shuffle(training_data)
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        def convert_to_arrays(data):
            return {
                'positions': np.array([sample['position'] for sample in data]),
                'additional_features': np.array([sample['additional_features'] for sample in data]),
                'move_from': np.array([sample['best_move_from'] for sample in data]),
                'move_to': np.array([sample['best_move_to'] for sample in data]),
                'evaluations': np.array([sample['position_eval'] for sample in data]),
                'game_phases': np.array([sample['game_phase'] for sample in data]),
            }
        
        train_dataset = convert_to_arrays(train_data)
        val_dataset = convert_to_arrays(val_data)
        
        return train_dataset, val_dataset
    
    def save_dataset(self, dataset: Dict, filename: str):
        """Save dataset to disk"""
        filepath = self.output_dir / filename
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filename: str) -> Dict:
        """Load dataset from disk"""
        filepath = self.output_dir / filename
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def create_training_pipeline(self, csv_file: str, max_games: int = 5000, 
                               min_rating: int = 1200):
        """Complete pipeline from CSV to training data"""
        
        # Process CSV file
        training_data = self.process_csv_games(csv_file, max_games=max_games, min_rating=min_rating)
        
        if not training_data:
            print("No training data extracted!")
            return None, None
        
        # Create datasets
        train_dataset, val_dataset = self.create_dataset(training_data)
        
        # Save datasets
        self.save_dataset(train_dataset, 'train_dataset.pkl.gz')
        self.save_dataset(val_dataset, 'val_dataset.pkl.gz')
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(train_dataset['positions'])}")
        print(f"Validation samples: {len(val_dataset['positions'])}")
        print(f"Position shape: {train_dataset['positions'][0].shape}")
        print(f"Additional features shape: {train_dataset['additional_features'][0].shape}")
        
        return train_dataset, val_dataset

def main():
    """Main processing script"""
    processor = ChessDataProcessor()
    
    # Process your CSV file
    csv_file = "games.csv"  # Your CSV file
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        print("Please make sure the games.csv file is in the current directory.")
        return
    
    print("🏛️  CSV Chess Data Pipeline")
    print("=" * 50)
    
    # Configuration
    max_games = int(input("Enter max number of games to process (default 5000): ") or "5000")
    min_rating = int(input("Enter minimum player rating (default 1200): ") or "1200")
    
    print(f"\nProcessing up to {max_games} games with minimum rating {min_rating}...")
    
    # Create training data
    train_data, val_data = processor.create_training_pipeline(
        csv_file=csv_file,
        max_games=max_games,
        min_rating=min_rating
    )
    
    if train_data is not None:
        print(f"\n✅ Success! Created dataset with {len(train_data['positions'])} training positions")
        print("You can now train your AI with: python chess_ai_model.py")
    else:
        print("\n❌ Failed to create dataset")

if __name__ == "__main__":
    main()