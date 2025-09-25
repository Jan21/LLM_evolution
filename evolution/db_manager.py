"""Database management module for experiment tracking."""

import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, Optional, Any
from omegaconf import DictConfig, OmegaConf
import inspect


class DatabaseManager:
    """Manages all database operations for experiment tracking."""
    
    def __init__(self, db_path: str = 'experiments.db'):
        self.db_path = db_path
        self._create_db()
    
    def _create_db(self):
        """Create SQLite database and experiments table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            d_model INTEGER,
            num_heads INTEGER,
            num_layers INTEGER,
            d_ff INTEGER,
            max_seq_length INTEGER,
            dropout REAL,
            learning_rate REAL,
            batch_size INTEGER,
            max_epochs INTEGER,
            weight_decay REAL,
            warmup_steps INTEGER,
            gradient_clip_val REAL,
            val_loss REAL,
            val_accuracy REAL,
            val_path_validity REAL,
            val_edge_accuracy REAL,
            val_exact_match_accuracy REAL,
            model_class_string TEXT,
            config_yaml TEXT,
            iteration INTEGER DEFAULT 0
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_current_iteration(self) -> int:
        """Get the current iteration number from database."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT MAX(iteration) FROM experiments')
                result = cursor.fetchone()
                conn.close()
                
                max_iteration = result[0] if result and result[0] is not None else -1
                return max_iteration + 1
                
            except sqlite3.Error as e:
                print(f"Database read attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return 0
                time.sleep(0.1)
    
    def store_experiment(self, cfg: DictConfig, results: Dict[str, float], 
                        model_class_str: str, iteration: Optional[int] = None) -> int:
        """Store experiment configuration and results in database."""
        if iteration is None:
            iteration = self.get_current_iteration()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO experiments (
                    timestamp, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,
                    learning_rate, batch_size, max_epochs, weight_decay, warmup_steps, gradient_clip_val,
                    val_loss, val_accuracy, val_path_validity, val_edge_accuracy, val_exact_match_accuracy,
                    model_class_string, config_yaml, iteration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    cfg.model.d_model,
                    cfg.model.num_heads,
                    cfg.model.num_layers,
                    cfg.model.d_ff,
                    cfg.model.max_seq_length,
                    cfg.model.dropout,
                    cfg.training.learning_rate,
                    cfg.training.batch_size,
                    cfg.training.max_epochs,
                    cfg.training.weight_decay,
                    cfg.training.warmup_steps,
                    cfg.training.gradient_clip_val,
                    float(results.get('val_loss', 0)),
                    float(results.get('val_accuracy', 0)),
                    float(results.get('val_path_validity', 0)),
                    float(results.get('val_edge_accuracy', 0)),
                    float(results.get('val_exact_match_accuracy', 0)),
                    model_class_str,
                    OmegaConf.to_yaml(cfg),
                    iteration
                ))
                
                experiment_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                print(f"Experiment stored in database with ID: {experiment_id}, iteration: {iteration}")
                return experiment_id
                
            except sqlite3.Error as e:
                print(f"Database insert attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1)
    
    def get_best_experiment(self, iteration: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best experiment (lowest val_loss) from database.
        
        Args:
            iteration: If specified, get best experiment from that iteration only.
                      If None, get best experiment across all iterations.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if iteration is not None:
                    cursor.execute('''
                    SELECT * FROM experiments 
                    WHERE iteration = ?
                    ORDER BY val_loss ASC 
                    LIMIT 1
                    ''', (iteration,))
                else:
                    cursor.execute('''
                    SELECT * FROM experiments 
                    ORDER BY val_loss ASC 
                    LIMIT 1
                    ''')
                
                result = cursor.fetchone()
                conn.close()
                
                if not result:
                    return None
                
                # Convert to dict
                column_names = [
                    'id', 'timestamp', 'd_model', 'num_heads', 'num_layers', 'd_ff', 
                    'max_seq_length', 'dropout', 'learning_rate', 'batch_size', 'max_epochs',
                    'weight_decay', 'warmup_steps', 'gradient_clip_val', 'val_loss', 
                    'val_accuracy', 'val_path_validity', 'val_edge_accuracy', 'val_exact_match_accuracy',
                    'model_class_string', 'config_yaml', 'iteration'
                ]
                
                return dict(zip(column_names, result))
                
            except sqlite3.Error as e:
                print(f"Database read attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.1)
    
    def get_experiments_by_iteration(self, iteration: int) -> list:
        """Get all experiments from a specific iteration."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM experiments 
            WHERE iteration = ?
            ORDER BY val_loss ASC
            ''', (iteration,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            column_names = [
                'id', 'timestamp', 'd_model', 'num_heads', 'num_layers', 'd_ff', 
                'max_seq_length', 'dropout', 'learning_rate', 'batch_size', 'max_epochs',
                'weight_decay', 'warmup_steps', 'gradient_clip_val', 'val_loss', 
                'val_accuracy', 'val_path_validity', 'val_edge_accuracy', 'val_exact_match_accuracy',
                'model_class_string', 'config_yaml', 'iteration'
            ]
            
            return [dict(zip(column_names, row)) for row in results]
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []