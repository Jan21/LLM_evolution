"""Model evolution module for LLM-based model generation and validation."""

import os
import sys
import tempfile
import importlib.util
import traceback
import inspect
import openai
import torch
from typing import Tuple, Optional, Dict, Any


class ModelEvolution:
    """Handles model evolution through LLM generation."""
    
    def __init__(self, api_key_file: str = 'api_key.txt'):
        self.api_key = self._load_api_key(api_key_file)
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _load_api_key(self, api_key_file: str) -> str:
        """Load OpenAI API key from file."""
        try:
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            return api_key
        except FileNotFoundError:
            raise FileNotFoundError(f"{api_key_file} file not found. Please create this file with your OpenAI API key.")
        except Exception as e:
            raise Exception(f"Error reading API key: {e}")
    
    def generate_evolved_model(self, best_experiment: Dict[str, Any], 
                             evolution_iteration: int, 
                             max_retries: int = 5) -> Tuple[str, str]:
        """
        Generate a new evolved model based on the best experiment.
        
        Args:
            best_experiment: Dictionary containing the best experiment details
            evolution_iteration: The current evolution iteration (NOT the experiment iteration)
            max_retries: Maximum number of generation attempts
            
        Returns:
            Tuple of (generated_code, class_name)
        """
        # Use evolution_iteration for naming, not the experiment's iteration
        class_name = f"GeneratedTransformerModel_v{evolution_iteration}"
        pos_encoding_class_name = f"PositionalEncoding_v{evolution_iteration}"
        
        print(f"Generating new model for evolution iteration {evolution_iteration}")
        print(f"Based on experiment {best_experiment['id']} (val_loss: {best_experiment['val_loss']:.4f})")
        
        original_model_class = best_experiment['model_class_string']
        
        for attempt in range(max_retries):
            print(f"Generation attempt {attempt + 1}/{max_retries}")
            
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

            prompt = self._create_evolution_prompt(
                best_experiment, 
                evolution_iteration, 
                class_name, 
                pos_encoding_class_name,
                original_model_class,
                retry_instruction
            )
            
            try:
                generated_code = self._call_llm(prompt)
                
                if self._validate_generated_code(generated_code, class_name, evolution_iteration):
                    print(f"✓ Successfully generated and validated new model class {class_name}")
                    return generated_code, class_name
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
    
    def _create_evolution_prompt(self, best_experiment: Dict[str, Any], 
                                evolution_iteration: int,
                                class_name: str,
                                pos_encoding_class_name: str,
                                original_model_class: str,
                                retry_instruction: str) -> str:
        """Create the prompt for the LLM."""
        return f"""You are an AI model architecture engineer. Your task is to evolve and improve a PyTorch transformer model based on the performance of a previous experiment.

{retry_instruction}

CONTEXT:
- This is evolution iteration {evolution_iteration} of an evolutionary model improvement process
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
- Add a print statement in __init__ showing the model is being used

OUTPUT FORMAT:
Return only the Python code for the two classes, starting with imports. No explanatory text before or after the code.

# Generated model iteration {evolution_iteration}
import torch
import torch.nn as nn
import math

[YOUR GENERATED CLASSES HERE]
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Make the API call to ChatGPT."""
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
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
            # temperature=0.7,
            # top_p=0.9
        )
        
        generated_code = response.choices[0].message.content.strip()
        
        # Clean up the response - remove any markdown formatting
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        elif generated_code.startswith("```"):
            generated_code = generated_code.replace("```", "").strip()
        
        print(f"Generated code length: {len(generated_code)} characters")
        return generated_code
    
    def _validate_generated_code(self, generated_code: str, class_name: str, iteration: int) -> bool:
        """Test if the generated code can be imported and instantiated."""
        temp_file = None
        try:
            # Write the generated code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                f.write(generated_code)
            
            # Try to load the module
            spec = importlib.util.spec_from_file_location("temp_model", temp_file)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            
            # Try to get the generated classes
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
                
            print(f"✓ Generated code validation passed for {class_name}")
            return True
            
        except Exception as e:
            print(f"✗ Generated code validation failed: {e}")
            return False
        finally:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def save_generated_model(self, model_code: str, class_name: str) -> None:
        """Save the generated model class to generated_models.py."""
        
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
        lines = model_code.split('\n')
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
            final_content = model_code
        else:
            # Combine: imports/helpers + existing classes + new class
            final_content = imports_and_helpers + existing_classes + '\n\n\n' + new_class_only
        
        # Write the combined content
        with open('generated_models.py', 'w') as f:
            f.write(final_content)
        
        print(f"Added new generated model {class_name} to generated_models.py")
    
    def get_model_class_string(self, class_name: str) -> str:
        """Get the source code for a model class."""
        if class_name == 'TransformerModel':
            from model.model import TransformerModel
            return inspect.getsource(TransformerModel)
        else:
            # Try to import from generated_models
            try:
                generated_module = importlib.import_module('generated_models')
                ModelClass = getattr(generated_module, class_name)
                return inspect.getsource(ModelClass)
            except (ImportError, AttributeError) as e:
                print(f"Failed to get source for {class_name}, using TransformerModel. Error: {e}")
                from model.model import TransformerModel
                return inspect.getsource(TransformerModel)