import pickle
import json
import random
import networkx as nx
from typing import List, Dict, Tuple
import hydra
from omegaconf import DictConfig
import os


def load_graph(graph_path: str) -> nx.Graph:
    """Load graph from pickle file"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def random_walk(graph: nx.Graph, start_node: int, length: int) -> List[int]:
    """Perform random walk from start_node for given length"""
    path = [start_node]
    current = start_node
    
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        
        # Remove previous node from neighbors to avoid going back
        if len(path) >= 2:
            previous = path[-2]
            if previous in neighbors:
                neighbors.remove(previous)
        
        # If no valid neighbors remain, break
        if not neighbors:
            break
            
        current = random.choice(neighbors)
        path.append(current)
    
    return path


def generate_train_data(graph: nx.Graph, num_paths: int, min_length: int, max_length: int) -> List[Dict]:
    """Generate training data with random walks"""
    train_data = []
    nodes = list(graph.nodes())
    
    for _ in range(num_paths):
        # Random path length
        path_length = random.randint(min_length, max_length)
        
        # Random starting node
        start_node = random.choice(nodes)
        
        # Perform random walk
        path = random_walk(graph, start_node, path_length)
        
        # Create training example
        if len(path) >= 2:
            train_example = {
                "input": [path[0], path[-1]],
                "output": path
            }
            train_data.append(train_example)
    
    return train_data


def generate_test_data(graph: nx.Graph, num_paths: int, train_pairs: set) -> List[Dict]:
    """Generate test data with shortest paths, avoiding train pairs"""
    test_data = []
    nodes = list(graph.nodes())
    attempts = 0
    max_attempts = num_paths * 10  # Prevent infinite loop
    
    while len(test_data) < num_paths and attempts < max_attempts:
        attempts += 1
        
        # Random pair of nodes
        start_node = random.choice(nodes)
        end_node = random.choice(nodes)
        
        # Skip if same node or pair already in training
        if start_node == end_node or (start_node, end_node) in train_pairs or (end_node, start_node) in train_pairs:
            continue
        
        # Find shortest path
        try:
            shortest_path = nx.shortest_path(graph, start_node, end_node)
            
            test_example = {
                "input": [start_node, end_node],
                "output": shortest_path
            }
            test_data.append(test_example)
            
        except nx.NetworkXNoPath:
            # No path exists between these nodes
            continue
    
    return test_data


@hydra.main(version_base=None, config_path="../config", config_name="config")
def generate_dataset(cfg: DictConfig) -> None:
    """Main function to generate training and test datasets"""
    
    # Load graph
    print(f"Loading graph from {cfg.data_generation.graph_path}")
    graph = load_graph(cfg.data_generation.graph_path)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Generate training data
    print(f"Generating {cfg.data_generation.train.num_paths} training examples...")
    train_data = generate_train_data(
        graph,
        cfg.data_generation.train.num_paths,
        cfg.data_generation.train.min_length,
        cfg.data_generation.train.max_length
    )
    
    # Extract training pairs for test set filtering
    train_pairs = set()
    for example in train_data:
        start, end = example["input"]
        train_pairs.add((start, end))
    
    # Generate test data
    print(f"Generating {cfg.data_generation.test.num_paths} test examples...")
    test_data = generate_test_data(
        graph,
        cfg.data_generation.test.num_paths,
        train_pairs
    )
    
    # Save datasets
    train_file = os.path.join(cfg.data_generation.output_dir, "train.json")
    test_file = os.path.join(cfg.data_generation.output_dir, "test.json")
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training data saved to {train_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved to {test_file}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    if train_data:
        train_lengths = [len(example["output"]) for example in train_data]
        print(f"Training path lengths: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.2f}")
    
    if test_data:
        test_lengths = [len(example["output"]) for example in test_data]
        print(f"Test path lengths: min={min(test_lengths)}, max={max(test_lengths)}, avg={sum(test_lengths)/len(test_lengths):.2f}")


if __name__ == "__main__":
    generate_dataset()