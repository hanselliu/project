import numpy as np
from qutip import rand_dm, partial_transpose

def is_ppt(state):
    """Return True if the given state has a positive partial transpose, and False otherwise."""
    pt = partial_transpose(state, [1,0]) # Perform a partial transpose on the second subsystem
    eigenvalues = pt.eigenenergies() # Get the eigenvalues of the partial transpose
    return np.all(eigenvalues >= 0) # Check if all eigenvalues are non-negative


def generate_states(n_samples):
    """Generate random two-qubit states and label them as entangled (1) or separable (0) based on the PPT criterion."""
    states = np.empty((n_samples, 16), dtype=np.complex) # 4*4 = 16 coefficients for a two-qubit density matrix
    labels = np.empty(n_samples, dtype=int)
    
    for i in range(n_samples):
        rho = rand_dm(4, density=0.75, dims=[[2,2],[2,2]]) # Generate a random density matrix
        states[i] = rho.full().flatten() # Flatten the density matrix into a 1D array
        labels[i] = 0 if is_ppt(rho) else 1 # Label the state based on the PPT criterion
    
    return states, labels
