"""
Chess AI Neural Network Model
Implements a deep learning model for chess move prediction and position evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import gzip
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm import tqdm

class ChessDataset(Dataset):
    """PyTorch Dataset for chess training data"""
    
    def __init__(self, dataset_path: str):
        """Load dataset from pickle file"""
        with gzip.open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Convert to tensors and fix dimensions
        # PyTorch expects (batch, channels, height, width) but we have (batch, height, width, channels)
        positions = torch.FloatTensor(self.data['positions'])
        self.positions = positions.permute(0, 3, 1, 2)  # Reshape to (batch, 12, 8, 8)
        
        self.additional_features = torch.FloatTensor(self.data['additional_features'])
        self.move_from = torch.LongTensor(self.data['move_from'])
        self.move_to = torch.LongTensor(self.data['move_to'])
        self.evaluations = torch.FloatTensor(self.data['evaluations'])
        
        print(f"Loaded dataset with {len(self.positions)} samples")
        print(f"Position shape: {self.positions.shape}")
        print(f"Additional features shape: {self.additional_features.shape}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'additional_features': self.additional_features[idx],
            'move_from': self.move_from[idx],
            'move_to': self.move_to[idx],
            'evaluation': self.evaluations[idx]
        }

class ChessConvBlock(nn.Module):
    """Convolutional block for chess position processing"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ChessNet(nn.Module):
    """Chess AI Neural Network
    
    Architecture:
    - Input: 8x8x12 board representation + additional features
    - Convolutional layers for spatial pattern recognition
    - Two heads: move prediction and position evaluation
    """
    
    def __init__(self, additional_features_dim: int = 8):
        super().__init__()
        
        # Convolutional backbone for board analysis
        self.conv1 = ChessConvBlock(12, 64)  # 12 input channels (6 pieces x 2 colors)
        self.conv2 = ChessConvBlock(64, 128)
        self.conv3 = ChessConvBlock(128, 256)
        self.conv4 = ChessConvBlock(256, 256)
        self.conv5 = ChessConvBlock(256, 256)
        
        # Global pooling and feature combination
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Additional features processing
        self.additional_fc = nn.Sequential(
            nn.Linear(additional_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined features dimension
        combined_dim = 256 + 64  # conv features + additional features
        
        # Move prediction head - predicts from/to squares
        self.move_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Separate outputs for from and to squares
        self.move_from_output = nn.Linear(256, 64)  # 64 squares
        self.move_to_output = nn.Linear(256, 64)    # 64 squares
        
        # Position evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, position, additional_features):
        # Process board position through convolutions
        x = self.conv1(position)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Global average pooling
        conv_features = self.global_pool(x).squeeze(-1).squeeze(-1)  # [batch, 256]
        
        # Process additional features
        additional_out = self.additional_fc(additional_features)
        
        # Combine features
        combined = torch.cat([conv_features, additional_out], dim=1)
        
        # Move prediction
        move_features = self.move_head(combined)
        move_from_logits = self.move_from_output(move_features)
        move_to_logits = self.move_to_output(move_features)
        
        # Position evaluation
        position_eval = self.eval_head(combined)
        
        return {
            'move_from': move_from_logits,
            'move_to': move_to_logits,
            'evaluation': position_eval.squeeze(-1)
        }

class ChessTrainer:
    """Training loop and utilities for chess AI"""
    
    def __init__(self, model: ChessNet, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.move_criterion = nn.CrossEntropyLoss()
        self.eval_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            position = batch['position'].to(self.device)
            additional_features = batch['additional_features'].to(self.device)
            move_from_target = batch['move_from'].to(self.device)
            move_to_target = batch['move_to'].to(self.device)
            eval_target = batch['evaluation'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(position, additional_features)
            
            # Calculate losses
            move_from_loss = self.move_criterion(outputs['move_from'], move_from_target)
            move_to_loss = self.move_criterion(outputs['move_to'], move_to_target)
            eval_loss = self.eval_criterion(outputs['evaluation'], eval_target)
            
            # Combined loss with weights
            total_batch_loss = move_from_loss + move_to_loss + 0.5 * eval_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'from_loss': f'{move_from_loss.item():.4f}',
                'to_loss': f'{move_to_loss.item():.4f}',
                'eval_loss': f'{eval_loss.item():.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move data to device
                position = batch['position'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                move_from_target = batch['move_from'].to(self.device)
                move_to_target = batch['move_to'].to(self.device)
                eval_target = batch['evaluation'].to(self.device)
                
                # Forward pass
                outputs = self.model(position, additional_features)
                
                # Calculate losses
                move_from_loss = self.move_criterion(outputs['move_from'], move_from_target)
                move_to_loss = self.move_criterion(outputs['move_to'], move_to_target)
                eval_loss = self.eval_criterion(outputs['evaluation'], eval_target)
                
                total_batch_loss = move_from_loss + move_to_loss + 0.5 * eval_loss
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 50):
        """Full training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_chess_model.pth')
                patience_counter = 0
                print("✓ New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_chess_model.pth'))
        print("Training completed! Best model loaded.")
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def predict_move(self, position: np.ndarray, additional_features: np.ndarray) -> Tuple[int, int]:
        """Predict best move for a given position"""
        self.model.eval()
        with torch.no_grad():
            # Convert from (8, 8, 12) to (1, 12, 8, 8)
            position_tensor = torch.FloatTensor(position).permute(2, 0, 1).unsqueeze(0).to(self.device)
            additional_tensor = torch.FloatTensor(additional_features).unsqueeze(0).to(self.device)
            
            outputs = self.model(position_tensor, additional_tensor)
            
            # Get predicted squares
            from_square = torch.argmax(outputs['move_from'], dim=1).item()
            to_square = torch.argmax(outputs['move_to'], dim=1).item()
            
            return from_square, to_square
    
    def evaluate_position(self, position: np.ndarray, additional_features: np.ndarray) -> float:
        """Evaluate a chess position"""
        self.model.eval()
        with torch.no_grad():
            # Convert from (8, 8, 12) to (1, 12, 8, 8)
            position_tensor = torch.FloatTensor(position).permute(2, 0, 1).unsqueeze(0).to(self.device)
            additional_tensor = torch.FloatTensor(additional_features).unsqueeze(0).to(self.device)
            
            outputs = self.model(position_tensor, additional_tensor)
            return outputs['evaluation'].item()

def main():
    """Main training script"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ChessDataset('chess_data/train_dataset.pkl.gz')
    val_dataset = ChessDataset('chess_data/val_dataset.pkl.gz')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model and trainer
    model = ChessNet(additional_features_dim=8)
    trainer = ChessTrainer(model, device)
    
    # Train the model
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot results
    trainer.plot_training_history()
    
    print("Training completed! Model saved as 'best_chess_model.pth'")

if __name__ == "__main__":
    main()