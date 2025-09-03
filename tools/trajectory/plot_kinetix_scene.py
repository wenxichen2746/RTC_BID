import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_scene(json_filepath):
    """
    Loads a scene from a JSON file and plots it using Matplotlib.

    Args:
        json_filepath (str): The path to the JSON file defining the scene.
    """
    # 1. Load the JSON data
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    env_state = data['env_state']
    
    # Define colors for different object roles to match the image
    role_colors = {
        0: '#4c4c4c',  # Dark Gray for walls, car wheels
        1: '#2ca02c',  # Green for car body
        2: '#1f77b4',  # Blue for the goal object
        3: '#8c564b',   # Brown/Red for vertical obstacles
    }

    # 2. Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#fdfbf6') # Set a background color similar to the image
    ax.set_aspect('equal', 'box')
    ax.set_title("2D Physics Environment Visualization")
    
    # 3. Plot Polygons
    if 'polygon' in env_state:
        for poly_data in env_state['polygon']:
            if not poly_data['active']:
                continue

            # Extract position, rotation, and vertices
            pos = np.array([poly_data['position']['0'], poly_data['position']['1']])
            angle = poly_data['rotation']
            
            # Convert vertex dictionary to a NumPy array
            local_vertices = np.array([[v['0'], v['1']] for v in poly_data['vertices'].values()])
            
            # Create the 2D rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            
            # Apply rotation and then translation to get world coordinates
            rotated_vertices = local_vertices @ rotation_matrix.T
            world_vertices = rotated_vertices + pos
            
            # Create a Matplotlib patch and add it to the plot
            role = poly_data.get('role', 0)
            color = role_colors.get(role, 'gray')
            polygon_patch = patches.Polygon(world_vertices, closed=True, facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(polygon_patch)

    # 4. Plot Circles
    if 'circle' in env_state:
        for circle_data in env_state['circle']:
            if not circle_data['active']:
                continue
            
            center = (circle_data['position']['0'], circle_data['position']['1'])
            radius = circle_data['radius']
            
            role = circle_data.get('role', 0)
            color = role_colors.get(role, 'gray')
            
            circle_patch = patches.Circle(center, radius, facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle_patch)
    
    # 5. Finalize and Display the Plot
    # Set plot limits based on the scene's content
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.grid(False) # Turn off the grid for a cleaner look
    plt.show()


if __name__ == "__main__":
    # The script assumes the JSON file is in the same directory
    plot_scene("./worlds/l/swing_up_hard_ver1.json")