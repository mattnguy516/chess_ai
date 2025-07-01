"""
Chess Engine Bridge - Converts between your chess engine and python-chess library
This bridges the gap between your custom chess engine and the standard python-chess format
used for AI training data processing.
"""

import chess
from chess_engine import ChessBoard, Color, PieceType, Piece, Move
from typing import Optional

class ChessEngineBridge:
    """Bridge between your chess engine and python-chess library"""
    
    def __init__(self):
        # Mapping from your piece types to python-chess piece types
        self.piece_type_mapping = {
            PieceType.PAWN: chess.PAWN,
            PieceType.ROOK: chess.ROOK,
            PieceType.KNIGHT: chess.KNIGHT,
            PieceType.BISHOP: chess.BISHOP,
            PieceType.QUEEN: chess.QUEEN,
            PieceType.KING: chess.KING,
        }
        
        # Reverse mapping
        self.reverse_piece_mapping = {v: k for k, v in self.piece_type_mapping.items()}
        
        # Color mapping
        self.color_mapping = {
            Color.WHITE: chess.WHITE,
            Color.BLACK: chess.BLACK,
        }
        
        self.reverse_color_mapping = {v: k for k, v in self.color_mapping.items()}
    
    def your_board_to_python_chess(self, your_board: ChessBoard) -> chess.Board:
        """Convert your ChessBoard to python-chess Board"""
        # Start with empty board
        python_board = chess.Board(fen=None)
        python_board.clear_board()
        
        # Copy pieces
        for row in range(8):
            for col in range(8):
                piece = your_board.get_piece(row, col)
                if piece:
                    # Convert to python-chess square (0-63)
                    square = (7 - row) * 8 + col  # Convert coordinate system
                    
                    # Convert piece type and color
                    python_piece_type = self.piece_type_mapping[piece.piece_type]
                    python_color = self.color_mapping[piece.color]
                    
                    # Create python-chess piece
                    python_piece = chess.Piece(python_piece_type, python_color)
                    python_board.set_piece_at(square, python_piece)
        
        # Set turn
        python_board.turn = self.color_mapping[your_board.current_player]
        
        # Set castling rights (simplified - you might need to track this better)
        python_board.castling_rights = chess.BB_EMPTY
        
        # Check if kings and rooks haven't moved for castling rights
        white_king = your_board.get_piece(7, 4)  # e1
        black_king = your_board.get_piece(0, 4)  # e8
        
        if white_king and white_king.piece_type == PieceType.KING and not white_king.has_moved:
            white_rook_h1 = your_board.get_piece(7, 7)  # h1
            white_rook_a1 = your_board.get_piece(7, 0)  # a1
            
            if white_rook_h1 and white_rook_h1.piece_type == PieceType.ROOK and not white_rook_h1.has_moved:
                python_board.castling_rights |= chess.BB_H1
            if white_rook_a1 and white_rook_a1.piece_type == PieceType.ROOK and not white_rook_a1.has_moved:
                python_board.castling_rights |= chess.BB_A1
        
        if black_king and black_king.piece_type == PieceType.KING and not black_king.has_moved:
            black_rook_h8 = your_board.get_piece(0, 7)  # h8
            black_rook_a8 = your_board.get_piece(0, 0)  # a8
            
            if black_rook_h8 and black_rook_h8.piece_type == PieceType.ROOK and not black_rook_h8.has_moved:
                python_board.castling_rights |= chess.BB_H8
            if black_rook_a8 and black_rook_a8.piece_type == PieceType.ROOK and not black_rook_a8.has_moved:
                python_board.castling_rights |= chess.BB_A8
        
        # Set en passant target
        if your_board.en_passant_target:
            ep_row, ep_col = your_board.en_passant_target
            ep_square = (7 - ep_row) * 8 + ep_col
            python_board.ep_square = ep_square
        else:
            python_board.ep_square = None
        
        return python_board
    
    def python_chess_to_your_board(self, python_board: chess.Board) -> ChessBoard:
        """Convert python-chess Board to your ChessBoard"""
        your_board = ChessBoard()
        
        # Clear the board first
        for row in range(8):
            for col in range(8):
                your_board.board[row][col] = None
        
        # Copy pieces
        for square in chess.SQUARES:
            piece = python_board.piece_at(square)
            if piece:
                # Convert square to your coordinate system
                row = 7 - (square // 8)
                col = square % 8
                
                # Convert piece type and color
                your_piece_type = self.reverse_piece_mapping[piece.piece_type]
                your_color = self.reverse_color_mapping[piece.color]
                
                # Create your piece
                your_piece = Piece(your_piece_type, your_color)
                your_board.board[row][col] = your_piece
        
        # Set current player
        your_board.current_player = self.reverse_color_mapping[python_board.turn]
        
        # Set castling rights (simplified)
        # You might want to track has_moved more carefully based on castling rights
        
        # Set en passant target
        if python_board.ep_square is not None:
            ep_row = 7 - (python_board.ep_square // 8)
            ep_col = python_board.ep_square % 8
            your_board.en_passant_target = (ep_row, ep_col)
        else:
            your_board.en_passant_target = None
        
        return your_board
    
    def your_move_to_python_chess(self, your_move: Move, python_board: chess.Board) -> Optional[chess.Move]:
        """Convert your Move to python-chess Move"""
        from_row, from_col = your_move.from_pos
        to_row, to_col = your_move.to_pos
        
        # Convert to python-chess squares
        from_square = (7 - from_row) * 8 + from_col
        to_square = (7 - to_row) * 8 + to_col
        
        # Handle promotion
        promotion = None
        if your_move.promotion:
            promotion = self.piece_type_mapping[your_move.promotion]
        
        try:
            python_move = chess.Move(from_square, to_square, promotion)
            # Verify the move is legal in the python-chess board
            if python_move in python_board.legal_moves:
                return python_move
            else:
                return None
        except:
            return None
    
    def python_chess_to_your_move(self, python_move: chess.Move, your_board: ChessBoard) -> Optional[Move]:
        """Convert python-chess Move to your Move"""
        # Convert squares to your coordinate system
        from_row = 7 - (python_move.from_square // 8)
        from_col = python_move.from_square % 8
        to_row = 7 - (python_move.to_square // 8)
        to_col = python_move.to_square % 8
        
        # Get the piece being moved
        piece = your_board.get_piece(from_row, from_col)
        if not piece:
            return None
        
        # Get captured piece if any
        captured_piece = your_board.get_piece(to_row, to_col)
        
        # Handle promotion
        promotion = None
        if python_move.promotion:
            promotion = self.reverse_piece_mapping[python_move.promotion]
        
        # Create your move
        your_move = Move(
            from_pos=(from_row, from_col),
            to_pos=(to_row, to_col),
            piece=piece,
            captured_piece=captured_piece,
            promotion=promotion
        )
        
        return your_move
    
    def square_to_coordinates(self, square: int) -> tuple:
        """Convert python-chess square (0-63) to (row, col) coordinates"""
        row = 7 - (square // 8)
        col = square % 8
        return (row, col)
    
    def coordinates_to_square(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to python-chess square (0-63)"""
        return (7 - row) * 8 + col

# Example usage and testing
if __name__ == "__main__":
    # Test the bridge
    bridge = ChessEngineBridge()
    
    # Create a test board with your engine
    your_board = ChessBoard()
    print("Your board:")
    your_board.display()
    
    # Convert to python-chess
    python_board = bridge.your_board_to_python_chess(your_board)
    print(f"\nPython-chess FEN: {python_board.fen()}")
    
    # Convert back
    converted_back = bridge.python_chess_to_your_board(python_board)
    print("\nConverted back:")
    converted_back.display()
    
    print("\nBridge test completed successfully!")