import numpy as np

def are_vectors_parallel(v1, v2, tolerance=1e-8):
    cross_product = np.linalg.norm(np.cross(v1, v2))
    return np.isclose(cross_product, 0, atol=tolerance)

def find_parallel_pairs_with_distance(line1, line2, tolerance=1e-8):
    A = np.array(line1)
    B = np.array(line2)

    # Create matrices to represent all possible combinations of segments
    A_matrix = np.tile(A[:, np.newaxis, :], (1, B.shape[0], 1))
    B_matrix = np.tile(B[np.newaxis, :, :], (A.shape[0], 1, 1))

    # Check for parallelism
    parallel_mask = np.apply_along_axis(lambda x: are_vectors_parallel(x[0], x[1], tolerance), -1, np.dstack((A_matrix, B_matrix)))

    # Find indices of parallel pairs
    parallel_indices = np.column_stack(np.where(parallel_mask))

    # Calculate distances between parallel segments
    distances = np.linalg.norm(A_matrix - B_matrix, axis=-1)[parallel_mask]

    # Combine indices and distances into a single array
    parallel_pairs_with_distance = list(zip(parallel_indices, distances))

    return parallel_pairs_with_distance

# Example usage:
line1_directions = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

line2_directions = [
    [2, 4, 6],
    [8, 10, 12],
    [14, 16, 18]
]

parallel_pairs_with_distance = find_parallel_pairs_with_distance(line1_directions, line2_directions)
print(parallel_pairs_with_distance)
