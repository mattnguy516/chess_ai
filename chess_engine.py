"""
Basic Chess Engine, handles chess rules, move validation, 
and game management
"""

import copy
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set

class Color(Enum):
    WHITE = "white"
    BLACK = "black"

class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"

class Piece:
    def __init__(self, piece_type: PieceType, color: Color):
        self.piece_type = piece_type
        self.color = color
        self.has_moved = False  # For castling and pawn double moves
    
    def __str__(self):
        symbols = {
            (PieceType.PAWN, Color.WHITE): '♙',
            (PieceType.ROOK, Color.WHITE): '♖',
            (PieceType.KNIGHT, Color.WHITE): '♘',
            (PieceType.BISHOP, Color.WHITE): '♗',
            (PieceType.QUEEN, Color.WHITE): '♕',
            (PieceType.KING, Color.WHITE): '♔',
            (PieceType.PAWN, Color.BLACK): '♟',
            (PieceType.ROOK, Color.BLACK): '♜',
            (PieceType.KNIGHT, Color.BLACK): '♞',
            (PieceType.BISHOP, Color.BLACK): '♝',
            (PieceType.QUEEN, Color.BLACK): '♛',
            (PieceType.KING, Color.BLACK): '♚',
        }
        return symbols.get((self.piece_type, self.color), '?')
    
    def __repr__(self):
        return f"{self.color.value.capitalize()} {self.piece_type.value.capitalize()}"

class Move:
    def __init__(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                 piece: Piece, captured_piece: Optional[Piece] = None,
                 promotion: Optional[PieceType] = None, is_castling: bool = False,
                 is_en_passant: bool = False):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece = piece
        self.captured_piece = captured_piece
        self.promotion = promotion
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
    
    def __str__(self):
        from_str = chr(ord('a') + self.from_pos[1]) + str(8 - self.from_pos[0])
        to_str = chr(ord('a') + self.to_pos[1]) + str(8 - self.to_pos[0])
        return f"{from_str}{to_str}"

class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Color.WHITE
        self.move_history = []
        self.en_passant_target = None  # Position where en passant capture is possible
        self.setup_initial_position()
    
    def setup_initial_position(self):
        """Set up the standard chess starting position"""
        # Place pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.BLACK)
            self.board[6][col] = Piece(PieceType.PAWN, Color.WHITE)
        
        # Place other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.BLACK)
            self.board[7][col] = Piece(piece_type, Color.WHITE)
    
    def display(self):
        """Display the chess board in a readable format"""
        print("  a b c d e f g h")
        print("  " + "-" * 16)
        for row in range(8):
            print(f"{8-row}|", end="")
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    print(f"{piece} ", end="")
                else:
                    print("· ", end="")
            print(f"|{8-row}")
        print("  " + "-" * 16)
        print("  a b c d e f g h")
        print(f"Current player: {self.current_player.value.capitalize()}")
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds"""
        return 0 <= row < 8 and 0 <= col < 8
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at given position"""
        if self.is_valid_position(row, col):
            return self.board[row][col]
        return None
    
    def is_enemy_piece(self, row: int, col: int, color: Color) -> bool:
        """Check if there's an enemy piece at given position"""
        piece = self.get_piece(row, col)
        return piece is not None and piece.color != color
    
    def is_friendly_piece(self, row: int, col: int, color: Color) -> bool:
        """Check if there's a friendly piece at given position"""
        piece = self.get_piece(row, col)
        return piece is not None and piece.color == color
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if position is empty"""
        return self.get_piece(row, col) is None
    
    def get_pawn_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid pawn moves"""
        moves = []
        direction = -1 if piece.color == Color.WHITE else 1
        start_row = 6 if piece.color == Color.WHITE else 1
        
        # Forward move
        new_row = row + direction
        if self.is_valid_position(new_row, col) and self.is_empty(new_row, col):
            # Check for promotion
            if new_row == 0 or new_row == 7:
                for promotion_piece in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                    moves.append(Move((row, col), (new_row, col), piece, promotion=promotion_piece))
            else:
                moves.append(Move((row, col), (new_row, col), piece))
            
            # Double move from starting position
            if row == start_row:
                new_row = row + 2 * direction
                if self.is_valid_position(new_row, col) and self.is_empty(new_row, col):
                    moves.append(Move((row, col), (new_row, col), piece))
        
        # Diagonal captures
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if self.is_valid_position(new_row, new_col):
                if self.is_enemy_piece(new_row, new_col, piece.color):
                    captured_piece = self.get_piece(new_row, new_col)
                    # Check for promotion
                    if new_row == 0 or new_row == 7:
                        for promotion_piece in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                            moves.append(Move((row, col), (new_row, new_col), piece, captured_piece, promotion_piece))
                    else:
                        moves.append(Move((row, col), (new_row, new_col), piece, captured_piece))
                
                # En passant
                elif self.en_passant_target == (new_row, new_col):
                    captured_piece = self.get_piece(row, new_col)  # The pawn to be captured
                    moves.append(Move((row, col), (new_row, new_col), piece, captured_piece, is_en_passant=True))
        
        return moves
    
    def get_rook_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid rook moves"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.is_valid_position(new_row, new_col):
                    break
                
                if self.is_empty(new_row, new_col):
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy_piece(new_row, new_col, piece.color):
                    captured_piece = self.get_piece(new_row, new_col)
                    moves.append(Move((row, col), (new_row, new_col), piece, captured_piece))
                    break
                else:  # Friendly piece
                    break
        
        return moves
    
    def get_bishop_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid bishop moves"""
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # diagonals
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.is_valid_position(new_row, new_col):
                    break
                
                if self.is_empty(new_row, new_col):
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy_piece(new_row, new_col, piece.color):
                    captured_piece = self.get_piece(new_row, new_col)
                    moves.append(Move((row, col), (new_row, new_col), piece, captured_piece))
                    break
                else:  # Friendly piece
                    break
        
        return moves
    
    def get_knight_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid knight moves"""
        moves = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                if self.is_empty(new_row, new_col):
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy_piece(new_row, new_col, piece.color):
                    captured_piece = self.get_piece(new_row, new_col)
                    moves.append(Move((row, col), (new_row, new_col), piece, captured_piece))
        
        return moves
    
    def get_queen_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid queen moves (combination of rook and bishop)"""
        return self.get_rook_moves(row, col, piece) + self.get_bishop_moves(row, col, piece)
    
    def get_king_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate all valid king moves"""
        moves = []
        king_directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in king_directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                if self.is_empty(new_row, new_col):
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy_piece(new_row, new_col, piece.color):
                    captured_piece = self.get_piece(new_row, new_col)
                    moves.append(Move((row, col), (new_row, new_col), piece, captured_piece))
        
        # Castling
        if not piece.has_moved:
            moves.extend(self.get_castling_moves(row, col, piece))
        
        return moves
    
    def get_castling_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        """Generate castling moves for the king"""
        moves = []
        
        # Kingside castling
        rook = self.get_piece(row, 7)
        if (rook and rook.piece_type == PieceType.ROOK and not rook.has_moved and
            self.is_empty(row, 5) and self.is_empty(row, 6)):
            if not self.is_in_check(piece.color):
                # Check if squares the king passes through are not under attack
                temp_board = copy.deepcopy(self)
                temp_board.board[row][5] = piece
                temp_board.board[row][col] = None
                if not temp_board.is_in_check(piece.color):
                    temp_board.board[row][6] = piece
                    temp_board.board[row][5] = None
                    if not temp_board.is_in_check(piece.color):
                        moves.append(Move((row, col), (row, 6), piece, is_castling=True))
        
        # Queenside castling
        rook = self.get_piece(row, 0)
        if (rook and rook.piece_type == PieceType.ROOK and not rook.has_moved and
            self.is_empty(row, 1) and self.is_empty(row, 2) and self.is_empty(row, 3)):
            if not self.is_in_check(piece.color):
                # Check if squares the king passes through are not under attack
                temp_board = copy.deepcopy(self)
                temp_board.board[row][3] = piece
                temp_board.board[row][col] = None
                if not temp_board.is_in_check(piece.color):
                    temp_board.board[row][2] = piece
                    temp_board.board[row][3] = None
                    if not temp_board.is_in_check(piece.color):
                        moves.append(Move((row, col), (row, 2), piece, is_castling=True))
        
        return moves
    
    def get_all_moves_for_piece(self, row: int, col: int) -> List[Move]:
        """Get all possible moves for a piece at given position"""
        piece = self.get_piece(row, col)
        if not piece or piece.color != self.current_player:
            return []
        
        move_generators = {
            PieceType.PAWN: self.get_pawn_moves,
            PieceType.ROOK: self.get_rook_moves,
            PieceType.KNIGHT: self.get_knight_moves,
            PieceType.BISHOP: self.get_bishop_moves,
            PieceType.QUEEN: self.get_queen_moves,
            PieceType.KING: self.get_king_moves,
        }
        
        return move_generators[piece.piece_type](row, col, piece)
    
    def get_all_legal_moves(self, color: Color = None) -> List[Move]:
        """Get all legal moves for the current player"""
        if color is None:
            color = self.current_player
        
        legal_moves = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    moves = self.get_all_moves_for_piece(row, col)
                    # Filter out moves that would put own king in check
                    for move in moves:
                        if self.is_legal_move(move):
                            legal_moves.append(move)
        
        return legal_moves
    
    def is_legal_move(self, move: Move) -> bool:
        """Check if a move is legal (doesn't put own king in check)"""
        # Make a temporary move
        temp_board = copy.deepcopy(self)
        temp_board.make_move(move, check_legality=False)
        
        # Check if this puts the current player's king in check
        return not temp_board.is_in_check(move.piece.color)
    
    def find_king(self, color: Color) -> Tuple[int, int]:
        """Find the position of the king for given color"""
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.piece_type == PieceType.KING and piece.color == color:
                    return (row, col)
        raise ValueError(f"King not found for {color}")
    
    def is_square_attacked(self, row: int, col: int, by_color: Color) -> bool:
        """Check if a square is attacked by pieces of given color"""
        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                if piece and piece.color == by_color:
                    # Get moves for this piece (but avoid infinite recursion for king)
                    if piece.piece_type == PieceType.KING:
                        # For king, just check adjacent squares
                        king_directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                        for dr, dc in king_directions:
                            if r + dr == row and c + dc == col:
                                return True
                    else:
                        moves = self.get_all_moves_for_piece(r, c)
                        for move in moves:
                            if move.to_pos == (row, col):
                                return True
        return False
    
    def is_in_check(self, color: Color) -> bool:
        """Check if the king of given color is in check"""
        king_pos = self.find_king(color)
        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        return self.is_square_attacked(king_pos[0], king_pos[1], opponent_color)
    
    def is_checkmate(self, color: Color) -> bool:
        """Check if the given color is in checkmate"""
        if not self.is_in_check(color):
            return False
        
        # If in check, see if there are any legal moves
        legal_moves = self.get_all_legal_moves(color)
        return len(legal_moves) == 0
    
    def is_stalemate(self, color: Color) -> bool:
        """Check if the given color is in stalemate"""
        if self.is_in_check(color):
            return False
        
        # If not in check, see if there are any legal moves
        legal_moves = self.get_all_legal_moves(color)
        return len(legal_moves) == 0
    
    def make_move(self, move: Move, check_legality: bool = True) -> bool:
        """Make a move on the board"""
        if check_legality and not self.is_legal_move(move):
            return False
        
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        
        # Handle en passant
        if move.is_en_passant:
            # Remove the captured pawn
            captured_pawn_row = from_row
            captured_pawn_col = to_col
            self.board[captured_pawn_row][captured_pawn_col] = None
        
        # Handle castling
        if move.is_castling:
            # Move the rook
            if to_col == 6:  # Kingside
                rook = self.board[from_row][7]
                self.board[from_row][7] = None
                self.board[from_row][5] = rook
                rook.has_moved = True
            else:  # Queenside
                rook = self.board[from_row][0]
                self.board[from_row][0] = None
                self.board[from_row][3] = rook
                rook.has_moved = True
        
        # Move the piece
        self.board[to_row][to_col] = move.piece
        self.board[from_row][from_col] = None
        
        # Handle promotion
        if move.promotion:
            self.board[to_row][to_col] = Piece(move.promotion, move.piece.color)
            self.board[to_row][to_col].has_moved = True
        else:
            move.piece.has_moved = True
        
        # Set en passant target
        self.en_passant_target = None
        if (move.piece.piece_type == PieceType.PAWN and 
            abs(from_row - to_row) == 2):
            self.en_passant_target = ((from_row + to_row) // 2, from_col)
        
        # Add to move history
        self.move_history.append(move)
        
        # Switch players
        self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        
        return True
    
    def parse_move(self, move_str: str) -> Optional[Move]:
        """Parse a move string like 'e2e4' into a Move object"""
        if len(move_str) < 4:
            return None
        
        try:
            from_col = ord(move_str[0].lower()) - ord('a')
            from_row = 8 - int(move_str[1])
            to_col = ord(move_str[2].lower()) - ord('a')
            to_row = 8 - int(move_str[3])
            
            piece = self.get_piece(from_row, from_col)
            if not piece:
                return None
            
            # Find the matching move from legal moves
            legal_moves = self.get_all_moves_for_piece(from_row, from_col)
            for move in legal_moves:
                if move.to_pos == (to_row, to_col):
                    # Handle promotion
                    if len(move_str) == 5:
                        promotion_map = {'q': PieceType.QUEEN, 'r': PieceType.ROOK, 
                                       'b': PieceType.BISHOP, 'n': PieceType.KNIGHT}
                        if move_str[4].lower() in promotion_map:
                            move.promotion = promotion_map[move_str[4].lower()]
                    return move
            
            return None
        except (ValueError, IndexError):
            return None

def main():
    """Main game loop for testing the chess engine"""
    board = ChessBoard()
    
    while True:
        board.display()
        
        # Check game end conditions
        if board.is_checkmate(board.current_player):
            winner = "Black" if board.current_player == Color.WHITE else "White"
            print(f"Checkmate! {winner} wins!")
            break
        elif board.is_stalemate(board.current_player):
            print("Stalemate! It's a draw!")
            break
        elif board.is_in_check(board.current_player):
            print(f"{board.current_player.value.capitalize()} is in check!")
        
        # Get player input
        move_input = input(f"Enter move for {board.current_player.value} (e.g., e2e4): ").strip()
        
        if move_input.lower() == 'quit':
            break
        
        # Parse and make move
        move = board.parse_move(move_input)
        if move and board.make_move(move):
            print(f"Move made: {move}")
        else:
            print("Invalid move! Try again.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()