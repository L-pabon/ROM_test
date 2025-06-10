import numpy as np
import jax.numpy as jnp
import pickle

def read_matrices(file_path):
    """
    Reads a text file containing B', Bhat, , and Msu matrices.
    
    Args:
    file_path (str): Path to the input text file.
    
    Returns:
    Bp (jnp.array): Array of 256 3x3 B' matrices.
    Bh (jnp.array): Array of 16 3x3 Bhat matrices.
    Bt (jnp.array): Array of 16 16x3 Btilde matrices.
    Msu (jnp.array): Array of 1 16x3 Msu matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Initialize empty lists to store matrices
    Bp_matrices = [[[0.0 for _ in range(3)] for _ in range(3)] for _ in range(16) for _ in range(16)]
    Bh_matrices = []
    Bt_matrices = []
    Msu_matrices = []
    
    # Initialize counters for each matrix type
    Bp_count = 0
    Bh_count = 0
    Bt_count = 0
    Msu_count = 0
    
    # Iterate over lines in the file
    for line in lines:
        # Check if the line contains a matrix label
        if "B'" in line:
            if line == "3 3\n" or line == "16 3\n":
                continue
            else:
                # Initialize an empty 3x3 matrix
                matrix = [[0.0 for _ in range(3)] for _ in range(3)]
                # Read the next 3 lines to fill the matrix
                for i in range(3):
                    values = list(map(float, lines[lines.index(line) + i + 2].strip().split()))
                    for j in range(3):
                        matrix[i][j] = values[j]
                # Append the matrix to the list
                Bp_matrices[Bp_count] = matrix
                Bp_count += 1
        elif "Bhat" in line:
            if line == "3 3\n" or line == "16 3\n":
                continue
            else:
                # Initialize an empty 3x3 matrix
                matrix = [[0.0 for _ in range(3)] for _ in range(3)]
                # Read the next 3 lines to fill the matrix
                for i in range(3):
                    values = list(map(float, lines[lines.index(line) + i + 2].strip().split()))
                    for j in range(3):
                        matrix[i][j] = values[j]
                # Append the matrix to the list
                Bh_matrices.append(matrix)
                Bh_count += 1
        elif "Btilde" in line:
            if line == "3 3\n" or line == "16 3\n":
                continue
            else:
                # Initialize an empty 16x3 matrix
                matrix = [[0.0 for _ in range(3)] for _ in range(16)]
                # Read the next 16 lines to fill the matrix
                for i in range(16):
                    values = list(map(float, lines[lines.index(line) + i + 2].strip().split()))
                    for j in range(3):
                        matrix[i][j] = values[j]
                # Append the matrix to the list
                Bt_matrices.append(matrix)
                Bt_count += 1
        elif "Msu" in line:
            if line == "3 3\n" or line == "16 3\n":
                continue
            else:
                # Initialize an empty 16x3 matrix
                matrix = [[0.0 for _ in range(3)] for _ in range(16)]
                # Read the next 16 lines to fill the matrix
                for i in range(16):
                    values = list(map(float, lines[lines.index(line) + i + 2].strip().split()))
                    for j in range(3):
                        matrix[i][j] = values[j]
                # Append the matrix to the list
                Msu_matrices.append(matrix)
                Msu_count += 1

    #Reshape Bp to be a 4th order tensor
    Bp_matrices = np.array(Bp_matrices).reshape(16,16,3,3)
    
    # Convert lists to JAX arrays
    Bp = jnp.array(Bp_matrices)
    Bh = jnp.array(Bh_matrices)
    Bt = jnp.array(Bt_matrices)
    Msu = jnp.array(Msu_matrices)
    
    return Bp, Bh, Bt, Msu

def create_B_matrices(Bp, Bh, Bt, Msu):
    """
    Creates a high-level matrix labeled "B_matrices" with the field names "Bp", "Bh", "Bt", and "Msu".
    
    Args:
    Bp (jnp.array): Array of 256 3x3 B' matrices.
    Bh (jnp.array): Array of 16 3x3 Bhat matrices.
    Bt (jnp.array): Array of 16 16x3 Btilde matrices.
    Msu (jnp.array): Array of 1 16x3 Msu matrix.
    
    Returns:
    B_matrices (dict): High-level matrix with the field names "Bp", "Bh", "Bt", and "Msu".
    """
    B_matrices = {
        "Bp": Bp,
        "Bh": Bh,
        "Bt": Bt,
        "Msu": Msu
    }
    
    return B_matrices

def export_to_pickle(B_matrices, output_file):
    """
    Exports the high-level matrix to a pickle file.
    
    Args:
    B_matrices (dict): High-level matrix with the field names "Bp", "Bh", "Bt", and "Msu".
    output_file (str): Path to the output pickle file.
    """
    with open(output_file, 'wb') as file:
        pickle.dump(B_matrices, file)

# Example usage
file_path = "modalflf.txt"
output_file = "modalflf.pkl"

Bp, Bh, Bt, Msu = read_matrices(file_path)
B_matrices = create_B_matrices(Bp, Bh, Bt, Msu)
export_to_pickle(B_matrices, output_file)
