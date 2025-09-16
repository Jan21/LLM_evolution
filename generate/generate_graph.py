import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import hydra
from omegaconf import DictConfig
import os

def generate_sphere_wire_mesh(num_horizontal=20, num_vertical=20):
    """
    Generate a wire mesh sphere graph with horizontal and vertical circles.
    
    Args:
        num_horizontal: Number of horizontal circles (latitude lines)
        num_vertical: Number of vertical circles (longitude lines)
    
    Returns:
        NetworkX graph representing the sphere wire mesh
    """
    G = nx.Graph()
    
    # Generate nodes
    nodes = []
    node_id = 0
    
    # Add north pole
    north_pole = node_id
    nodes.append((0, 0, 1))  # (x, y, z)
    G.add_node(node_id, pos=(0, 0, 1))
    node_id += 1
    
    # Add nodes for horizontal circles (excluding poles)
    for i in range(1, num_horizontal):
        # Latitude angle from north pole
        theta = np.pi * i / num_horizontal
        z = np.cos(theta)
        r = np.sin(theta)  # radius at this height
        
        # Add nodes around this circle
        circle_nodes = []
        for j in range(num_vertical):
            phi = 2 * np.pi * j / num_vertical
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            nodes.append((x, y, z))
            G.add_node(node_id, pos=(x, y, z))
            circle_nodes.append(node_id)
            node_id += 1
    
    # Add south pole
    south_pole = node_id
    nodes.append((0, 0, -1))
    G.add_node(node_id, pos=(0, 0, -1))
    node_id += 1
    
    # Add horizontal edges (latitude circles)
    current_node = 1  # Start after north pole
    
    for i in range(1, num_horizontal):
        # Connect nodes in this horizontal circle
        for j in range(num_vertical):
            current = current_node + j
            next_node = current_node + (j + 1) % num_vertical
            G.add_edge(current, next_node)
        current_node += num_vertical
    
    # Add vertical edges (longitude circles)
    for j in range(num_vertical):
        # Connect north pole to first ring
        first_ring_node = 1 + j
        G.add_edge(north_pole, first_ring_node)
        
        # Connect between horizontal rings
        for i in range(1, num_horizontal - 1):
            current = 1 + (i - 1) * num_vertical + j
            next_ring = 1 + i * num_vertical + j
            G.add_edge(current, next_ring)
        
        # Connect last ring to south pole
        last_ring_node = 1 + (num_horizontal - 2) * num_vertical + j
        G.add_edge(last_ring_node, south_pole)
    
    return G

def visualize_sphere_mesh(G):
    """Visualize the sphere wire mesh in 3D"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Extract coordinates
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    
    # Plot edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=0.5)
    
    # Plot nodes
    ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sphere Wire Mesh Graph')
    
    # Make the plot look more spherical
    max_range = 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.show()

def print_graph_stats(G):
    """Print statistics about the generated graph"""
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(G)}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def generate_graph(cfg: DictConfig) -> None:
    """Main function to generate and save graph"""
    
    # Generate the sphere wire mesh
    print(f"Generating sphere mesh with {cfg.graph_generation.sphere_mesh.num_horizontal} horizontal and {cfg.graph_generation.sphere_mesh.num_vertical} vertical circles")
    sphere_graph = generate_sphere_wire_mesh(
        num_horizontal=cfg.graph_generation.sphere_mesh.num_horizontal,
        num_vertical=cfg.graph_generation.sphere_mesh.num_vertical
    )
    
    # Print statistics if requested
    if cfg.graph_generation.output.print_stats:
        print_graph_stats(sphere_graph)
    
    # Create output directory if it doesn't exist
    output_path = cfg.graph_generation.output.file_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save graph to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(sphere_graph, f)
    print(f"Graph saved to {output_path}")
    
    # Visualize the graph if requested
    if cfg.graph_generation.output.visualize:
        visualize_sphere_mesh(sphere_graph)


if __name__ == "__main__":
    generate_graph()