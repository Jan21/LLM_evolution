import sqlite3
import json
import inspect
import os
import sys
from datetime import datetime
from omegaconf import OmegaConf
from model.model import TransformerModel
import time

import openai
import os
import tempfile
import importlib.util
import sys
import traceback

def validate_generated_code(generated_class, class_name, iteration):
    """Test if the generated code can be imported and instantiated"""
    try:
        # Write the generated code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_file = f.name
            f.write(generated_class)
        
        # Try to load the module
        spec = importlib.util.spec_from_file_location("temp_model", temp_file)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Try to get the generated class
        ModelClass = getattr(temp_module, class_name)
        PositionalEncodingClass = getattr(temp_module, f"PositionalEncoding_v{iteration}")
        
        # Try to instantiate the model with basic parameters
        model = ModelClass(
            vocab_size=1000,
            d_model=64,
            num_heads=8,
            num_layers=2,
            d_ff=256,
            max_seq_length=32,
            dropout=0.1
        )
        
        # Try a simple forward pass
        import torch
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            
        # Check output shape
        expected_shape = (batch_size, seq_len, 1000)  # vocab_size
        if output.shape != expected_shape:
            raise ValueError(f"Output shape {output.shape} doesn't match expected {expected_shape}")
            
        # Clean up temp file
        os.unlink(temp_file)
        
        print(f"✓ Generated code validation passed for {class_name}")
        return True
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)
        
        print(f"✗ Generated code validation failed: {e}")
        return False

def load_api_key():
    """Load OpenAI API key from api_key.txt file"""
    try:
        with open('api_key.txt', 'r') as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError("api_key.txt file not found. Please create this file with your OpenAI API key.")
    except Exception as e:
        raise Exception(f"Error reading API key: {e}")

def llm_generate_model(best_experiment, max_retries=5):
    """Call ChatGPT to generate new model class based on best experiment with retry logic"""
    print(f"Generating new model based on experiment {best_experiment['id']} (val_loss: {best_experiment['val_loss']:.4f})")
    
    # Load API key and set up OpenAI client
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key)
    
    iteration = best_experiment.get('iteration', 0) + 1
    class_name = f"GeneratedTransformerModel_v{iteration}"
    pos_encoding_class_name = f"PositionalEncoding_v{iteration}"
    
    # Extract the original model class string from the best experiment
    original_model_class = best_experiment['model_class_string']
    
    for attempt in range(max_retries):
        print(f"Generation attempt {attempt + 1}/{max_retries}")
        
        # Create the prompt for the LLM
        retry_instruction = ""
        if attempt > 0:
            retry_instruction = f"""
IMPORTANT: This is attempt {attempt + 1} because previous attempts failed to run correctly. 
Please be extra careful about:
- Correct indentation and syntax
- Proper import statements
- Matching method signatures exactly
- Valid tensor operations
- Correct PyTorch module structure
"""

        prompt = f"""You are an AI model architecture engineer. Your task is to evolve and improve a PyTorch transformer model based on the performance of a previous experiment.

{retry_instruction}

CONTEXT:
- This is iteration {iteration} of an evolutionary model improvement process
- The previous best model had a validation loss of {best_experiment['val_loss']:.4f}
- Model hyperparameters from best experiment:
  - d_model: {best_experiment['d_model']}
  - num_heads: {best_experiment['num_heads']}
  - num_layers: {best_experiment['num_layers']}
  - d_ff: {best_experiment['d_ff']}
  - dropout: {best_experiment['dropout']}
  - learning_rate: {best_experiment['learning_rate']}

ORIGINAL MODEL CODE:
{original_model_class}

TASK:
Generate an improved version of this transformer model. You should:
1. Keep the same overall structure and interface
2. Make intelligent improvements that could reduce validation loss
3. Consider improvements like:
   - Different activation functions (GELU, Swish, etc.)
   - Layer normalization placement (pre-norm vs post-norm)
   - Different initialization strategies
   - Additional regularization techniques
   - Architectural modifications (residual connections, etc.)
   - Attention mechanism improvements

REQUIREMENTS:
- Generate exactly TWO classes: {pos_encoding_class_name} and {class_name}
- Keep the same method signatures and interfaces exactly
- Add comments explaining your evolutionary improvements
- The model must be compatible with the existing training code
- Include proper imports (torch, torch.nn, math)
- Ensure all tensor operations are valid and shapes match
- Use proper PyTorch module inheritance

OUTPUT FORMAT:
Return only the Python code for the two classes, starting with imports. No explanatory text before or after the code.

# Generated model iteration {iteration}
import torch
import torch.nn as nn
import math

[YOUR GENERATED CLASSES HERE]
"""

        try:
            # Make the API call to ChatGPT
            response = client.chat.completions.create(
                model="gpt-5-mini",  # Using ChatGPT-4o-mini as requested
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert AI model architect specializing in transformer improvements. Generate only clean, working PyTorch code without any explanatory text. Ensure the code is syntactically correct and will run without errors."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_completion_tokens=16384,
                # temperature=0.3 + (attempt * 0.1),  # Increase creativity with retries
                # top_p=0.9
            )
            
            generated_class = response.choices[0].message.content.strip()
            # print("Raw response repr:", repr(response))
            # Clean up the response - remove any markdown formatting if present
            if generated_class.startswith("```python"):
                generated_class = generated_class.replace("```python", "").replace("```", "").strip()
            elif generated_class.startswith("```"):
                generated_class = generated_class.replace("```", "").strip()
            
            print(f"Generated code length: {len(generated_class)} characters")
            if validate_generated_code(generated_class, class_name, iteration):
                print(f"✓ Successfully generated and validated new model class {class_name} using ChatGPT")
                return generated_class, class_name
            else:
                print(f"✗ Generated code failed validation, retrying...")
                continue
        
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting 60 seconds before retry...")
            import time
            time.sleep(60)
            continue
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate working code after {max_retries} attempts due to API errors")
            continue
        except Exception as e:
            print(f"Error during generation attempt {attempt + 1}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate working code after {max_retries} attempts")
            continue
    
    raise Exception(f"Failed to generate working code after {max_retries} attempts")


# Replace the original mock function with the new LLM function
def mock_llm_generate_model(best_experiment):
    """Wrapper to maintain compatibility with existing code"""
    return llm_generate_model(best_experiment)

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


def save_generated_model(model_class_string, class_name):
    """Save the generated model class to generated_models.py"""
    
    # Check if file exists and read current content
    existing_content = ""
    imports_and_helpers = ""
    existing_classes = ""
    
    if os.path.exists('generated_models.py'):
        with open('generated_models.py', 'r') as f:
            existing_content = f.read()
            
        # Split existing content to separate imports/helpers from classes
        if existing_content:
            lines = existing_content.split('\n')
            in_class = False
            current_section = []
            
            for line in lines:
                if line.startswith('class '):
                    if not in_class:
                        imports_and_helpers = '\n'.join(current_section) + '\n\n'
                        current_section = []
                        in_class = True
                    current_section.append(line)
                else:
                    current_section.append(line)
                    
            if in_class:
                existing_classes = '\n'.join(current_section)
    
    # Extract just the new class from the generated string
    lines = model_class_string.split('\n')
    new_class_lines = []
    in_new_class = False
    
    for line in lines:
        if line.startswith('class '):
            in_new_class = True
        if in_new_class:
            new_class_lines.append(line)
    
    new_class_only = '\n'.join(new_class_lines)
    
    # If no existing content, use the full generated string
    if not existing_content:
        final_content = model_class_string
    else:
        # Combine: imports/helpers + existing classes + new class
        final_content = imports_and_helpers + existing_classes + '\n\n\n' + new_class_only
    
    # Write the combined content
    with open('generated_models.py', 'w') as f:
        f.write(final_content)
    
    print(f"Added new generated model {class_name} to generated_models.py")


def run_multirun_experiment(model_class_name=None, log_file="multirun.log"):
    """Run multirun hyperparameter sweep and store results in a log file"""
    import subprocess
    import sys
    from datetime import datetime

    # Create/open log file
    with open(log_file, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Experiment started at {datetime.now()}\n")
        f.write("="*60 + "\n")

        cmd = [sys.executable, 'train.py', '--multirun']
        if model_class_name:
            cmd.append(f'model.class_name={model_class_name}')

        f.write(f"Running command: {' '.join(cmd)}\n")
        f.flush()

        # Redirect stdout and stderr to the log file
        result = subprocess.run(
            cmd,
            cwd='.',
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

        if result.returncode != 0:
            f.write(f"Multirun failed with return code: {result.returncode}\n")
            return None

        f.write("Multirun completed successfully!\n")
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