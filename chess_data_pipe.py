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
    
    def download_sample_games(self) -> str:
        """Download a smaller sample of games for testing"""
        # Alternative: Use a smaller dataset for testing
        # You can download sample PGN files from:
        # - https://www.pgnmentor.com/files.html
        # - https://github.com/niklasf/python-chess/tree/master/examples
        
        sample_pgn = """
[Event "Sample Game 1"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Bb7 10. d4 Re8 11. Nbd2 Bf8 12. a4 h6 13. Bc2 exd4 14. cxd4 Nb4 15. Bb1 c5 16. d5 Nd7 17. Ra3 f5 18. exf5 Rxe1+ 19. Nxe1 Bxd5 20. Nd3 Nxd3 21. Bxd3 Ne5 22. Be4 Bxe4 23. Nxe4 Re8 24. Re3 Nf7 25. axb5 axb5 26. Ng5 hxg5 27. Bxg5 Nxg5 28. h4 Ne4 29. Re1 Nf6 30. f3 Re3 31. Rxe3 1-0

[Event "Sample Game 2"]
[Site "Test"]
[Date "2024.01.01"]
[Round "2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[WhiteElo "1900"]
[BlackElo "1850"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Nd2 a5 10. Rb1 Nd7 11. a3 f5 12. b4 Kh8 13. f3 Ng8 14. Qc2 Ngf6 15. Nb5 axb4 16. axb4 Nh5 17. g3 Nhf6 18. c5 dxc5 19. bxc5 Nxc5 20. Nxc7 Rb8 21. Nb5 Qe7 22. Be3 fxe4 23. fxe4 Nfxe4 24. Nxe4 Nxe4 25. Bf3 Nf6 26. Rfc1 Bd7 27. Nc3 Rfc8 28. Qd3 Rxc3 29. Rxc3 Rc8 30. Rbc1 Rxc3 31. Rxc3 Qe1+ 32. Kg2 Qe2+ 0-1
"""
        
        filepath = self.output_dir / "sample_games.pgn"
        with open(filepath, 'w') as f:
            f.write(sample_pgn)
        
        print(f"Created sample PGN file: {filepath}")
        return str(filepath)
    
    def download_lichess_data(self, year: int = 2024, month: int = 12) -> str:
        """Download Lichess database for a specific month"""
        url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
        filename = f"lichess_{year}_{month:02d}.pgn.zst"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            print(f"File {filename} already exists")
            return str(filepath)
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        return str(filepath)
    
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
    
    def move_to_target(self, move: chess.Move) -> Tuple[int, int]:
        """Convert move to from/to square indices"""
        return move.from_square, move.to_square
    
    def evaluate_position(self, board: chess.Board, game_result: str) -> float:
        """Simple position evaluation based on game outcome"""
        if game_result == "1-0":  # White wins
            return 1.0 if board.turn == chess.WHITE else -1.0
        elif game_result == "0-1":  # Black wins
            return -1.0 if board.turn == chess.WHITE else 1.0
        else:  # Draw
            return 0.0
    
    def process_pgn_file(self, pgn_file: str, max_games: int = 10000, 
                        min_elo: int = 1500) -> List[Dict]:
        """Process PGN file and extract training data"""
        training_data = []
        games_processed = 0
        
        print(f"Processing {pgn_file}...")
        
        # Handle compressed files
        if pgn_file.endswith('.bz2'):
            import bz2
            file_obj = bz2.open(pgn_file, 'rt', encoding='utf-8')
        elif pgn_file.endswith('.gz'):
            file_obj = gzip.open(pgn_file, 'rt', encoding='utf-8')
        elif pgn_file.endswith('.zst'):
            try:
                import zstandard as zstd
                with open(pgn_file, 'rb') as compressed_file:
                    dctx = zstd.ZstdDecompressor()
                    stream_reader = dctx.stream_reader(compressed_file)
                    file_obj = io.TextIOWrapper(stream_reader, encoding='utf-8')
            except ImportError:
                print("Error: zstandard library not installed. Please install with: pip install zstandard")
                return []
        else:
            file_obj = open(pgn_file, 'r', encoding='utf-8')
        
        try:
            with file_obj as f:
                while games_processed < max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Filter by ELO if available
                    try:
                        white_elo = int(game.headers.get("WhiteElo", "0"))
                        black_elo = int(game.headers.get("BlackElo", "0"))
                        if white_elo < min_elo or black_elo < min_elo:
                            continue
                    except ValueError:
                        continue
                    
                    # Get game result
                    result = game.headers.get("Result", "*")
                    if result == "*":  # Skip unfinished games
                        continue
                    
                    # Process moves
                    board = game.board()
                    moves = list(game.mainline_moves())
                    
                    # Skip very short games
                    if len(moves) < 10:
                        continue
                    
                    # Extract positions from the game
                    for i, move in enumerate(moves):
                        # Skip opening moves (first 6 moves)
                        if i < 6:
                            board.push(move)
                            continue
                        
                        # Skip endgame (last 10 moves) to focus on middlegame
                        if i >= len(moves) - 10:
                            break
                        
                        # Create training sample
                        position_features = self.board_to_features(board)
                        additional_features = self.get_additional_features(board)
                        
                        # Get the best move (the one actually played)
                        from_square, to_square = self.move_to_target(move)
                        
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
                    
                    games_processed += 1
                    if games_processed % 100 == 0:
                        print(f"Processed {games_processed} games, {len(training_data)} positions")
        
        except Exception as e:
            print(f"Error processing file: {e}")
        
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
    
    def create_training_pipeline(self, pgn_file: str = None, 
                               max_games: int = 10000, use_sample: bool = False):
        """Complete pipeline from PGN to training data"""
        
        # Download data if no file provided
        if pgn_file is None:
            if use_sample:
                pgn_file = self.download_sample_games()
            else:
                # Try to download Lichess data
                try:
                    pgn_file = self.download_lichess_data(2024, 12)  # Download January 2024
                except Exception as e:
                    print(f"Failed to download Lichess data: {e}")
                    print("Using sample games instead...")
                    pgn_file = self.download_sample_games()
        
        # Process PGN file
        training_data = self.process_pgn_file(pgn_file, max_games=max_games)
        
        if not training_data:
            print("No training data extracted!")
            return
        
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

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = ChessDataProcessor()
    
    # Option 1: Use sample data for testing (recommended first)
    # print("Testing with sample data...")
    # train_data, val_data = processor.create_training_pipeline(max_games=10, use_sample=True)
    
    # Option 2: Download real Lichess data (requires zstandard: pip install zstandard)
    train_data, val_data = processor.create_training_pipeline(max_games=1000)
    
    # Option 3: Process your own PGN file:
    # train_data, val_data = processor.create_training_pipeline("your_games.pgn", max_games=1000)