import sqlite3
import pandas as pd
import sys


def view_experiments(limit=10, sort_by='val_loss'):
    """View stored experiments from the database"""
    try:
        conn = sqlite3.connect('experiments.db')
        
        # Get all experiments (excluding model_class_string and config_yaml for readability)
        query = f'''
        SELECT 
            id, timestamp, d_model, num_heads, num_layers, d_ff, 
            learning_rate, batch_size, val_loss, val_accuracy, 
            val_path_validity, val_edge_accuracy, val_exact_match_accuracy
        FROM experiments 
        ORDER BY {sort_by} ASC
        LIMIT {limit}
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No experiments found in database.")
            return
        
        print(f"Top {limit} experiments (sorted by {sort_by}):")
        print("=" * 100)
        print(df.to_string(index=False))
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def view_best_experiment():
    """View the best experiment based on validation loss"""
    try:
        conn = sqlite3.connect('experiments.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM experiments 
        ORDER BY val_loss ASC 
        LIMIT 1
        ''')
        
        result = cursor.fetchone()
        
        if not result:
            print("No experiments found in database.")
            return
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        print("Best Experiment (lowest validation loss):")
        print("=" * 50)
        for col, val in zip(column_names, result):
            if col == 'model_class_string':
                print(f"{col}: [Model class code - {len(val)} characters]")
            elif col == 'config_yaml':
                print(f"{col}: [Config YAML - {len(val)} characters]")
            else:
                print(f"{col}: {val}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")


def get_model_class(experiment_id):
    """Get the model class string for a specific experiment"""
    try:
        conn = sqlite3.connect('experiments.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT model_class_string FROM experiments WHERE id = ?', (experiment_id,))
        result = cursor.fetchone()
        
        if result:
            print(f"Model class for experiment {experiment_id}:")
            print("=" * 50)
            print(result[0])
        else:
            print(f"No experiment found with ID {experiment_id}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'best':
            view_best_experiment()
        elif sys.argv[1] == 'model' and len(sys.argv) > 2:
            try:
                exp_id = int(sys.argv[2])
                get_model_class(exp_id)
            except ValueError:
                print("Please provide a valid experiment ID")
        elif sys.argv[1].isdigit():
            limit = int(sys.argv[1])
            sort_by = sys.argv[2] if len(sys.argv) > 2 else 'val_loss'
            view_experiments(limit, sort_by)
        else:
            print("Usage: python view_experiments.py [limit] [sort_by] | best | model <id>")
    else:
        view_experiments()