import sqlite3
import json
import inspect
import os
import sys
from datetime import datetime
from omegaconf import OmegaConf
from model.model import TransformerModel
import time


def create_db():
    """Create SQLite database and experiments table"""
    conn = sqlite3.connect('experiments.db')
    cursor = conn.cursor()
    
    # Use IF NOT EXISTS and handle potential race conditions
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
    return conn


def get_model_class_string():
    """Extract the TransformerModel class definition as a string"""
    return inspect.getsource(TransformerModel)


def get_current_iteration():
    """Get the current iteration number from database"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('experiments.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT MAX(iteration) FROM experiments')
            result = cursor.fetchone()
            conn.close()
            
            max_iteration = result[0] if result and result[0] is not None else -1
            return max_iteration + 1
            
        except sqlite3.Error as e:
            print(f"Database read attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return 0  # Default if all fails
            time.sleep(0.1)


def store_experiment_result(cfg, results, iteration=None):
    """Store experiment configuration and results in database"""
    if iteration is None:
        iteration = get_current_iteration()
    
    # Ensure database exists - handle concurrent access
    conn = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = create_db()
            break
        except sqlite3.Error as e:
            print(f"Database creation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1)
    
    cursor = conn.cursor()
    
    # Get model class string from the currently active model
    try:
        class_name = getattr(cfg.model, 'class_name', 'TransformerModel')
        if class_name != 'TransformerModel':
            import importlib
            generated_module = importlib.import_module('generated_models')
            ModelClass = getattr(generated_module, class_name)
            model_class_str = inspect.getsource(ModelClass)
        else:
            model_class_str = get_model_class_string()
    except (ImportError, AttributeError):
        model_class_str = get_model_class_string()
    
    # Insert with retry logic for concurrent access
    for attempt in range(max_retries):
        try:
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
            
            conn.commit()
            print(f"Experiment stored in database with ID: {cursor.lastrowid}")
            break
            
        except sqlite3.Error as e:
            print(f"Database insert attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1)
    
    conn.close()


def get_best_experiment():
    """Get the best experiment (lowest val_loss) from database"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('experiments.db')
            cursor = conn.cursor()
            
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


def mock_llm_generate_model(best_experiment):
    """Mock LLM call to generate new model class based on best experiment"""
    print(f"Generating new model based on experiment {best_experiment['id']} (val_loss: {best_experiment['val_loss']:.4f})")
    
    # Mock: Generate slight variations of the TransformerModel
    # In reality, this would call an LLM API
    iteration = best_experiment.get('iteration', 0) + 1
    class_name = f"GeneratedTransformerModel_v{iteration}"
    
    generated_class = f'''# Generated model iteration {iteration}
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 128):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class {class_name}(nn.Module):
    """Generated model iteration {iteration} - evolved from val_loss {best_experiment['val_loss']:.4f}"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',  # Could be evolved to 'gelu' or other activations
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Evolution: Add an extra layer or change initialization
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Maybe add layer norm or other improvements
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Evolution: different initialization strategy
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * (1 + {iteration} * 0.1))  
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Convert attention_mask to the format expected by transformer
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # transformer expects: 0 for real tokens, -inf for masked
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        x = self.transformer(
            x, 
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )
        
        # Evolution: add layer norm before output
        x = self.final_layer_norm(x)
        
        logits = self.output_projection(x)
        return logits
'''
    
    return generated_class, class_name


def save_generated_model(model_class_string, class_name):
    """Save the generated model class to generated_models.py"""
    with open('generated_models.py', 'w') as f:
        f.write(model_class_string)
    print(f"Saved new generated model {class_name} to generated_models.py")


def run_multirun_experiment(model_class_name=None):
    """Run multirun hyperparameter sweep and store results"""
    import subprocess
    import sys
    
    print("Starting multirun hyperparameter sweep...")
    
    cmd = [sys.executable, 'train.py', '--multirun']
    if model_class_name:
        cmd.append(f'model.class_name={model_class_name}')
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Stream output in real-time instead of capturing it
    result = subprocess.run(cmd, cwd='.')
    
    if result.returncode != 0:
        print(f"Multirun failed with return code: {result.returncode}")
        return None
    
    print("Multirun completed successfully!")
    return True


def evolutionary_training_loop(max_iterations=5):
    """Main evolutionary training loop"""
    print(f"Starting evolutionary training loop (max {max_iterations} iterations)")
    
    current_model_class = None  # Start with original model
    
    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*50}")
        
        # Run multirun training
        print(f"Running training iteration {iteration + 1}...")
        if current_model_class:
            print(f"Using generated model class: {current_model_class}")
        else:
            print("Using original TransformerModel")
            
        result = run_multirun_experiment(current_model_class)
        
        if not result:
            print(f"Training failed at iteration {iteration + 1}")
            break
        
        # Get best experiment from this iteration
        best_exp = get_best_experiment()
        
        if not best_exp:
            print("No experiments found in database!")
            break
            
        print(f"Best model from this iteration: ID {best_exp['id']}, val_loss: {best_exp['val_loss']:.4f}")
        
        # Don't generate new model on last iteration
        if iteration < max_iterations - 1:
            # Generate new model based on best performer
            new_model_class, class_name = mock_llm_generate_model(best_exp)
            
            # Save the generated model
            save_generated_model(new_model_class, class_name)
            current_model_class = class_name
            
            print(f"Generated new model {class_name} for iteration {iteration + 2}")
        else:
            print("Final iteration completed!")
    
    print("\nEvolutionary training completed!")
    
    # Show final best model
    final_best = get_best_experiment()
    if final_best:
        print(f"Overall best model: ID {final_best['id']}, val_loss: {final_best['val_loss']:.4f}")


def run_single_experiment():
    """Run a single experiment and store results"""
    import subprocess
    import sys
    
    print("Starting single training...")
    
    # Stream output in real-time instead of capturing it
    result = subprocess.run([sys.executable, 'train.py'], 
                          cwd='.')
    
    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        return None
    
    print("Training completed successfully!")
    return True


def run_with_custom_config(config_overrides=None, multirun=False):
    """Run experiment with custom configuration overrides"""
    import subprocess
    import sys
    
    cmd = [sys.executable, 'train.py']
    if multirun:
        cmd.append('--multirun')
    
    if config_overrides:
        for key, value in config_overrides.items():
            cmd.append(f'{key}={value}')
    
    print(f"Running with command: {' '.join(cmd)}")
    
    # Stream output in real-time instead of capturing it
    result = subprocess.run(cmd, cwd='.')
    
    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        return None
    
    print("Training completed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'evolve':
            # python run.py evolve [max_iterations]
            max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            evolutionary_training_loop(max_iter)
            print("Evolution completed!")
        elif sys.argv[1] == 'multirun':
            # python run.py multirun
            result = run_multirun_experiment()
            print(f"Multirun completed: {result}")
        elif sys.argv[1] == 'custom':
            # Example: python run.py custom model.d_model=128 model.num_heads=8
            overrides = {}
            multirun = False
            for arg in sys.argv[2:]:
                if arg == '--multirun':
                    multirun = True
                elif '=' in arg:
                    key, value = arg.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                    overrides[key] = value
            
            result = run_with_custom_config(overrides, multirun)
            print(f"Custom run completed: {result}")
        else:
            print("Usage: python run.py [evolve|multirun|custom] [args...]")
            sys.exit(1)
    else:
        result = run_single_experiment()
        print(f"Single experiment completed: {result}")