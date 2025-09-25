"""Main orchestration module for evolutionary model training."""

import sys
from omegaconf import DictConfig
from database_manager import DatabaseManager
from model_evolution import ModelEvolution
from experiment_runner import ExperimentRunner


def store_experiment_result(cfg: DictConfig, results: dict, iteration: int = None):
    """Store experiment results in the database."""
    db = DatabaseManager()
    
    # Get model class string
    evolution = ModelEvolution()
    class_name = getattr(cfg.model, 'class_name', 'TransformerModel')
    model_class_str = evolution.get_model_class_string(class_name)
    
    return db.store_experiment(cfg, results, model_class_str, iteration)


class EvolutionaryTrainer:
    """Orchestrates the evolutionary training loop."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.evolution = ModelEvolution()
        self.runner = ExperimentRunner()
    
    def run_evolutionary_loop(self, max_iterations: int = 5):
        """
        Main evolutionary training loop.
        
        Args:
            max_iterations: Maximum number of evolution iterations
        """
        print(f"Starting evolutionary training loop (max {max_iterations} iterations)")
        print("="*60)
        
        current_model_class = None  # Start with original model
        
        for evolution_iter in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"EVOLUTION ITERATION {evolution_iter + 1}/{max_iterations}")
            print(f"{'='*50}")
            
            # Determine the actual iteration number for database storage
            db_iteration = self.db.get_current_iteration()
            print(f"Database iteration: {db_iteration}")
            
            # Run multirun training
            if current_model_class:
                print(f"Using generated model class: {current_model_class}")
            else:
                print("Using original TransformerModel")
            
            success = self.runner.run_multirun_experiment(
                model_class_name=current_model_class,
                iteration=evolution_iter + 1
            )
            
            if not success:
                print(f"Training failed at evolution iteration {evolution_iter + 1}")
                break
            
            # Get best experiment from ALL iterations (not just current)
            best_overall = self.db.get_best_experiment()
            
            # Also show best from current iteration for comparison
            best_current_iter = self.db.get_best_experiment(iteration=db_iteration)
            
            if not best_overall:
                print("No experiments found in database!")
                break
            
            print(f"\nBest model overall: ID {best_overall['id']}, "
                  f"val_loss: {best_overall['val_loss']:.4f}, "
                  f"from iteration: {best_overall['iteration']}")
            
            if best_current_iter:
                print(f"Best model from current iteration {db_iteration}: "
                      f"ID {best_current_iter['id']}, "
                      f"val_loss: {best_current_iter['val_loss']:.4f}")
            
            # Don't generate new model on last iteration
            if evolution_iter < max_iterations - 1:
                # Generate new model based on best performer overall
                # BUT use the correct evolution iteration for naming
                new_model_code, class_name = self.evolution.generate_evolved_model(
                    best_overall, 
                    evolution_iteration=evolution_iter + 2  # Next iteration number
                )
                
                # Save the generated model
                self.evolution.save_generated_model(new_model_code, class_name)
                current_model_class = class_name
                
                print(f"Generated new model {class_name} for evolution iteration {evolution_iter + 2}")
                print(f"This model will be used in the next round of experiments")
            else:
                print("Final iteration completed!")
        
        print("\n" + "="*60)
        print("EVOLUTIONARY TRAINING COMPLETED!")
        print("="*60)
        
        # Show final best model
        final_best = self.db.get_best_experiment()
        if final_best:
            print(f"\nOverall best model across all iterations:")
            print(f"  ID: {final_best['id']}")
            print(f"  Iteration: {final_best['iteration']}")
            print(f"  Val Loss: {final_best['val_loss']:.4f}")
            print(f"  Val Accuracy: {final_best['val_accuracy']:.4f}")
            print(f"  Path Validity: {final_best['val_path_validity']:.4f}")
            print(f"  Edge Accuracy: {final_best['val_edge_accuracy']:.4f}")
        
        # Show summary statistics per iteration
        print("\n" + "="*60)
        print("SUMMARY BY ITERATION:")
        print("="*60)
        
        for iter_num in range(self.db.get_current_iteration()):
            experiments = self.db.get_experiments_by_iteration(iter_num)
            if experiments:
                best = experiments[0]  # Already sorted by val_loss
                avg_loss = sum(e['val_loss'] for e in experiments) / len(experiments)
                print(f"Iteration {iter_num}: "
                      f"{len(experiments)} experiments, "
                      f"best loss: {best['val_loss']:.4f}, "
                      f"avg loss: {avg_loss:.4f}")


def main():
    """Main entry point for the script."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'evolve':
            # python run.py evolve [max_iterations]
            max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            trainer = EvolutionaryTrainer()
            trainer.run_evolutionary_loop(max_iter)
            
        elif command == 'single':
            # python run.py single [config overrides...]
            runner = ExperimentRunner()
            config_overrides = {}
            
            for arg in sys.argv[2:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                    config_overrides[key] = value
            
            result = runner.run_single_experiment(config_overrides)
            print(f"Single experiment completed: {result}")
            
        elif command == 'multirun':
            # python run.py multirun [model_class_name]
            model_class = sys.argv[2] if len(sys.argv) > 2 else None
            runner = ExperimentRunner()
            result = runner.run_multirun_experiment(model_class)
            print(f"Multirun completed: {result}")
            
        elif command == 'custom':
            # python run.py custom [--multirun] [config overrides...]
            runner = ExperimentRunner()
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
            
            result = runner.run_with_custom_config(overrides, multirun)
            print(f"Custom run completed: {result}")
            
        else:
            print_usage()
            
    else:
        # Default: run single experiment
        runner = ExperimentRunner()
        result = runner.run_single_experiment()
        print(f"Single experiment completed: {result}")


def print_usage():
    """Print usage information."""
    print("Usage: python run.py [command] [args...]")
    print()
    print("Commands:")
    print("  evolve [max_iterations]    - Run evolutionary training loop")
    print("  single [config...]         - Run single experiment")
    print("  multirun [model_class]     - Run multirun sweep")
    print("  custom [--multirun] [config...] - Run with custom config")
    print()
    print("Examples:")
    print("  python run.py evolve 5")
    print("  python run.py single model.d_model=128")
    print("  python run.py multirun GeneratedTransformerModel_v2")
    print("  python run.py custom --multirun model.d_model=256 model.num_heads=8")
    sys.exit(1)


if __name__ == "__main__":
    main()