import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, Any
import wandb
import pickle
import networkx as nx

from .model import TransformerModel


class PathPredictionModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        graph_path: str = "temp/sphere_mesh_graph.pkl"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Load the graph for path validation
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def compute_loss(self, logits, targets):
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len) with -100 for padding tokens
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        return loss
    
    def compute_metrics(self, logits, targets):
        # Compute accuracy
        predictions = torch.argmax(logits, dim=-1)
        
        # Only compute accuracy on non-padded tokens (targets != -100)
        mask = (targets != -100)
        correct = (predictions == targets) & mask
        total = mask.sum()
        accuracy = correct.sum().float() / total.float() if total > 0 else torch.tensor(0.0)
        
        # Compute exact match accuracy (whole sequence correct)
        batch_size = predictions.size(0)
        exact_matches = 0
        for i in range(batch_size):
            # Only consider non-padded tokens for this example
            example_mask = mask[i]
            if example_mask.sum() > 0:  # Only if there are non-padded tokens
                example_correct = (predictions[i] == targets[i]) & example_mask
                if example_correct.sum() == example_mask.sum():
                    exact_matches += 1
        
        exact_match_accuracy = torch.tensor(exact_matches / batch_size, dtype=torch.float32, device=predictions.device)
        
        # Compute path validity and optimality
        path_validity = self._compute_path_validity(predictions, targets)
        edge_accuracy = self._compute_edge_accuracy(predictions, targets)
        
        return {
            'accuracy': accuracy,
            'exact_match_accuracy': exact_match_accuracy,
            'path_validity': path_validity,
            'edge_accuracy': edge_accuracy
        }
    
    def _compute_path_validity(self, predictions, targets):
        """Check if predicted paths contain valid edges in the graph"""
        batch_size = predictions.size(0)
        valid_paths = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens (where targets != -100)
            mask = targets[i] != -100
            path = predictions[i][mask]
            
            # Convert to list and remove padding tokens (assuming 0 is padding)
            path_list = path.cpu().tolist()[:-1] # remove the eos token
            path_list = [node for node in path_list if node != 0]
            
            if len(path_list) <= 1:
                valid_paths += 1  # Single node or empty path is valid
                continue
            
            # Check if consecutive nodes are connected
            is_valid = True
            for j in range(len(path_list) - 1):
                node1, node2 = path_list[j], path_list[j + 1]
                if not self.graph.has_edge(node1, node2):
                    is_valid = False
                    break
            
            if is_valid:
                valid_paths += 1
        
        return torch.tensor(valid_paths / batch_size, dtype=torch.float32, device=predictions.device)
    
    
    def _compute_edge_accuracy(self, predictions, targets):
        """Compute the proportion of predicted edges that are valid (exist in the graph)"""
        batch_size = predictions.size(0)
        total_valid_edges = 0
        total_edges = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens (where targets != -100)
            mask = targets[i] != -100
            pred_path = predictions[i][mask]
            
            # Convert to list and remove padding tokens
            pred_list = [node for node in pred_path.cpu().tolist()][:-1] # remove the eos token
            
            # Skip if path has less than 2 nodes (no edges)
            if len(pred_list) < 2:
                continue
            
            # Extract edges from predicted path
            pred_edges = [(pred_list[j], pred_list[j+1]) for j in range(len(pred_list)-1)]
            
            # Count valid edges (edges that exist in the graph)
            valid_edges = 0
            for edge in pred_edges:
                node1, node2 = edge
                if self.graph.has_edge(node1, node2):
                    valid_edges += 1
            
            total_valid_edges += valid_edges
            total_edges += len(pred_edges)
        
        if total_edges == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=predictions.device)
        
        return torch.tensor(total_valid_edges / total_edges, dtype=torch.float32, device=predictions.device)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        logits = self(input_ids)
        loss = self.compute_loss(logits, target_ids)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        logits = self(input_ids)
        loss = self.compute_loss(logits, target_ids)
        
        metrics = self.compute_metrics(logits, target_ids)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log('val_path_validity', metrics['path_validity'], on_epoch=True, prog_bar=True)
        self.log('val_edge_accuracy', metrics['edge_accuracy'], on_epoch=True, prog_bar=True)
        self.log('val_exact_match_accuracy', metrics['exact_match_accuracy'], on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Simple warmup + cosine annealing scheduler - good default for language modeling
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(progress, 1.0))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }