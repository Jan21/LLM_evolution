"""Experiment runner module for executing training runs."""

import subprocess
import sys
from datetime import datetime
from typing import Optional, Dict, Any


class ExperimentRunner:
    """Handles execution of training experiments."""
    
    def __init__(self, log_file: str = "multirun.log"):
        self.log_file = log_file
    
    def run_single_experiment(self, config_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """
        Run a single training experiment.
        
        Args:
            config_overrides: Optional dictionary of configuration overrides
            
        Returns:
            True if successful, False otherwise
        """
        print("Starting single training...")
        
        cmd = [sys.executable, 'train.py']
        
        if config_overrides:
            for key, value in config_overrides.items():
                cmd.append(f'{key}={value}')
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Stream output in real-time
        result = subprocess.run(cmd, cwd='.')
        
        if result.returncode != 0:
            print(f"Training failed with return code: {result.returncode}")
            return False
        
        print("Training completed successfully!")
        return True
    
    def run_multirun_experiment(self, model_class_name: Optional[str] = None, 
                              iteration: int = 0) -> bool:
        """
        Run multirun hyperparameter sweep.
        
        Args:
            model_class_name: Name of the model class to use (None for default)
            iteration: Current iteration number for logging
            
        Returns:
            True if successful, False otherwise
        """
        # Create/append to log file
        with open(self.log_file, "a") as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Multirun Experiment - Iteration {iteration}\n")
            f.write(f"Started at {datetime.now()}\n")
            f.write("="*60 + "\n")

            cmd = [sys.executable, 'train.py', '--multirun']
            
            if model_class_name:
                cmd.append(f'model.class_name={model_class_name}')
                f.write(f"Using model class: {model_class_name}\n")
            else:
                f.write("Using default TransformerModel\n")

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
                return False

            f.write("Multirun completed successfully!\n")
            f.write(f"Ended at {datetime.now()}\n")
            return True
    
    def run_with_custom_config(self, config_overrides: Dict[str, Any], 
                             multirun: bool = False) -> bool:
        """
        Run experiment with custom configuration overrides.
        
        Args:
            config_overrides: Dictionary of configuration overrides
            multirun: Whether to run as multirun
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [sys.executable, 'train.py']
        
        if multirun:
            cmd.append('--multirun')
        
        for key, value in config_overrides.items():
            cmd.append(f'{key}={value}')
        
        print(f"Running with command: {' '.join(cmd)}")
        
        # Stream output in real-time
        result = subprocess.run(cmd, cwd='.')
        
        if result.returncode != 0:
            print(f"Training failed with return code: {result.returncode}")
            return False
        
        print("Training completed successfully!")
        return True