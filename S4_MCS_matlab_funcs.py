#Functions used in S4_MCS_matlab

import matlab.engine
import os
import numpy as np
import gc

#MATLAB implementation
def run_MCS(matlab_folder, mcs_file) -> None:
    file_path = os.path.join(matlab_folder, mcs_file)
    eng = matlab.engine.start_matlab()  # Start MATLAB session
    eng.run(file_path, nargout=0)  # \Run the Matlab script to develop the PC state matrix

#Python implementation
# Function to open the lambda_per_year and mu_per_year matrices and develop TTR and TTF matrices with failures estimated using the inverse exponential distribution
def f_MCS_IHG(tts, lambdas, mus):
    lambdas = np.asarray(lambdas, dtype=float).flatten() #load the lambda matrix
    mus = np.asarray(mus, dtype=float).flatten() #load the mu matrix
    if lambdas.ndim != 1 or mus.ndim != 1:
        raise ValueError("lambdas and mus must be 1-dimensional arrays.") #ensure lambda and mu matrix are 1-D array
    U1 = np.random.rand(tts, len(lambdas)) #generate random numbers over the timesteps in the MCS
    U2 = np.random.rand(tts, len(mus)) #generate random numbers over the timesteps in the MCS
    lambda_matrix = lambdas[np.newaxis, :]  # Broadcast lambda for vectorized computation with shape (1, n)
    mu_matrix = mus[np.newaxis, :]          # Broadcast mu for vectorized computation with shape (1, n)
    with np.errstate(divide='ignore'): # Avoid divide-by-zero with masking
        TTF = -np.log(U1) / lambda_matrix #inverse exponential probability distribution
        TTF[:, lambdas == 0] = np.inf
        TTR = -np.log(U2) / mu_matrix
        TTR[:, mus == 0] = np.inf
    return TTF, TTR


# Function to simulate TTF and TTR over the simulation time and develop the failure state matrix B.
# For each failure in B due to a failed state in TTF, the failures last for the corresponding time in TTR to develop D
# B matrix has the simulated failures, D matrix has the simulated failures lasting for the corresponding repair time
# In each timestep with a failed state, a load flow calculation is done to identify which customers are interrupted
def f_MCS_IHG_and_process(tts, lambdas, mus):
    TTF, TTR = f_MCS_IHG(tts, lambdas, mus)
    mask_failure = TTF < 1 # Identify failures (TTF < 1)
    TTR_rounded = np.zeros_like(TTF, dtype=np.uint8) # Round TTR at those failure points
    TTR_rounded[mask_failure] = np.round(TTR[mask_failure]).astype(np.uint8) #Find duration of faults
    D = np.where(mask_failure, TTR_rounded, 1).astype(np.uint8) # Initialize D and assign durations
    for col in range(D.shape[1]):
        failure_times = np.flatnonzero(D[:, col] > 1)
        for start in failure_times:
            duration = D[start, col]
            end = min(start + duration, tts)
            D[start+1:end, col] = 0
    del TTF, TTR, TTR_rounded, mask_failure  # Free memory explicitly
    gc.collect()

    return D

