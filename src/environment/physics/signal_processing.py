"""
Signal processing primitives for RIS Duplex environment.

Includes:
 - Rician fading channel generation (LoS + NLoS) with optional Bjornson scaling
 - Geometric path-loss-like gains for transmitter–RIS–receiver links
 - Array response and LoS/NLoS components
 - Interference/noise aggregations (Gamma terms) for SINR calculation
 - Unit conversions and synthetic signal generators

Better Comments legend:
 - TODO: future improvements or refactors (non-functional here)
 - NOTE: important behavior or design intent
 - !: important runtime remark
 - ?: questioning a choice or highlighting an assumption
"""
import numpy as np

# ?  ----------------------------------------------     FUNCTIONS TO CALCULATE THE CHANNELS     --------------------------------------------------------


def rician_fading_channel(transmitter_position: np.ndarray, receiver_position: np.ndarray, 
                          W_h_t: int, W_h_r: int,
                          d_h_tx: float, d_h_rx: float, lambda_h: float, epsilon_h: float,
                          numpy_generator : np.random._generator.Generator,
                          loop_channel_mode: bool = False, bjornson = False, nlos_only = False, los_only = False,
                          ):
    """
    Combines LoS and NLoS components using defined functions below to compute the Rician fading channel. 
    Returns a matrix of size N_r X N_t with N_r the number of receiving antennas and N_t the number of transmitting 
    antennas.

    Parameters:
        transmitter_position (np.ndarray): A 2D array [x, y] representing the transmitter position.
        receiver_position (np.ndarray): A 2D array [x, y] representing the receiver position.
        W_h_t (int): Number of elements in the ULA at the transmitter.
        W_h_r (int): Number of elements in the ULA at the receiver.
        d_h_tx (float): Inter-element spacing in the ULA at the transmitter.
        d_h_rx (float): Inter-element spacing in the ULA at the receiver.
        lambda_h (float): Carrier wavelength.
        epsilon_h (float): Rician factor (ratio of power between LoS and NLoS components).
        numpy_generator (numpy.random._generator.Generator): generator used to draw new random values with numpy.

    Returns:
        np.ndarray: The combined Rician fading channel matrix.
    """
    if not loop_channel_mode:
        diff = receiver_position - transmitter_position
        distance = np.linalg.norm(diff) + 1e-12  # Avoid division by zero

        if bjornson:
            beta = ( (lambda_h**2) /  ((4 * np.pi) **2)  )  / (distance**2)
            #? Using the formula from Bjornson & Demir (2024), page 252. The book is well explained and detailed
        else:
            beta = 1
        
        if los_only:
            h_los = computing_LoS_2D(transmitter_position, receiver_position, W_h_t, W_h_r, d_h_tx, d_h_rx, lambda_h) # * Compute the deterministic LoS component
            return (np.sqrt(beta) * ( np.sqrt(epsilon_h / (epsilon_h + 1)) * h_los) )
        
        elif nlos_only:
            # * Compute the random NLoS component
            h_nlos = computing_NLoS(W_h_t = W_h_t, W_h_r = W_h_r, numpy_generator = numpy_generator) # * Compute the random NLoS component
            return np.sqrt(beta) *  np.sqrt(1 / (epsilon_h + 1)) * h_nlos
        
        else:
            h_los = computing_LoS_2D(transmitter_position, receiver_position, W_h_t, W_h_r, d_h_tx, d_h_rx, lambda_h)
            h_nlos = computing_NLoS(W_h_t = W_h_t, W_h_r = W_h_r, numpy_generator = numpy_generator)
            return (np.sqrt(beta) * ( np.sqrt(epsilon_h / (epsilon_h + 1)) * h_los + np.sqrt(1 / (epsilon_h + 1)) * h_nlos )) #? Testing the formula from Bjornson & Demir (2024) for LoS signal, page 252.

        # * Combine LoS and NLoS components to form the Rician fading channel

    else:
        # TODO: to be modified for dealing with self interference
        # Compute the path loss coefficient
        diff = receiver_position - transmitter_position
        distance = np.linalg.norm(diff) + 3e-2  # Avoid division by zero
        """alpha_h = -30  - 10 * alpha * np.log10(distance/1)
        alpha_h_linear = 10 ** (alpha_h / 10)"""

        if bjornson:
            beta = ( (lambda_h**2) /  ((4 * np.pi) **2)  )  / (distance**2)
            #? Testing the formula from Bjornson & Demir (2024), page 252. The book is well explained and detailed
        else:
            beta = 1 
        # Compute the deterministic LoS component
        h_los = computing_LoS_2D(transmitter_position, receiver_position, W_h_t, W_h_r, d_h_tx, d_h_rx, lambda_h)
        # Compute the random NLoS component
        h_nlos = computing_NLoS(W_h_t = W_h_t, W_h_r = W_h_r, numpy_generator = numpy_generator)
        
        if los_only:
            h_los = computing_LoS_2D(transmitter_position, receiver_position, W_h_t, W_h_r, d_h_tx, d_h_rx, lambda_h) # * Compute the deterministic LoS component
            return (np.sqrt(beta) * ( np.sqrt(epsilon_h / (epsilon_h + 1)) * h_los) )
        
        elif nlos_only:
            # * Compute the random NLoS component
            h_nlos = computing_NLoS(W_h_t = W_h_t, W_h_r = W_h_r, numpy_generator = numpy_generator) # * Compute the random NLoS component
            return np.sqrt(beta) *  np.sqrt(1 / (epsilon_h + 1)) * h_nlos
        
        else:
            h_los = computing_LoS_2D(transmitter_position, receiver_position, W_h_t, W_h_r, d_h_tx, d_h_rx, lambda_h)
            h_nlos = computing_NLoS(W_h_t = W_h_t, W_h_r = W_h_r, numpy_generator = numpy_generator)
            return (np.sqrt(beta) * ( np.sqrt(epsilon_h / (epsilon_h + 1)) * h_los + np.sqrt(1 / (epsilon_h + 1)) * h_nlos )) #? Testing the formula from Bjornson & Demir (2024) for LoS signal, page 252.



def calculate_gain_transmitter_ris_receiver(transmitter_position, receiver_position, RIS_position, RIS_Cells, lambda_h ):

    # TODO add the antenna gains for the transmitter and the receiver, according to Bjornson et Al 2024 " Introduction to Multiple Antenna Communications and Reconfigurable Surfaces" page 627-628.
    
    A_m = (lambda_h/4)**2
    d_t = np.linalg.norm(transmitter_position - RIS_position)
    d_r = np.linalg.norm(receiver_position - RIS_position)
    beta_t = (  A_m / (4 * np.pi * (d_t **2) ) )  

    beta_r = ( A_m / (4 * np.pi * (d_r ** 2) ) )
    return RIS_Cells ** 2 * beta_t * beta_r
    

# Define array response for ULA
def array_response(W_h: int, theta: float, d_h: float, lambda_h: float):
    """
    Compute the array response vector for a ULA.

    Parameters:
        W_h (int): Number of elements in the ULA.
        theta (float): Angle in radians (AoA or AoD).
        d_h (float): Inter-element spacing for the ULA.
    
    Returns:
        np.ndarray: Array response vector.
    """
    indices = np.arange(W_h)
    
    return np.exp(-1j * 2 * np.pi * d_h / lambda_h * indices * np.sin(theta)).reshape(-1, 1) # ? Array response provided in Bjornson & Demir (2024), pages 227 & 253 


def computing_LoS_2D(transmitter_position: np.ndarray, receiver_position: np.ndarray, 
                     W_h_t: int, W_h_r: int, d_h_tx: float, d_h_rx: float, lambda_h: float):
    """
    Computes the Line-of-Sight (LoS) component of the channel in a 2D setting.

    Parameters:
        transmitter_position (np.ndarray): A 2D array [x, y] representing the transmitter position.
        receiver_position (np.ndarray): A 2D array [x, y] representing the receiver position.
        W_h_t (int): Number of elements in the ULA at the transmitter.
        W_h_r (int): Number of elements in the ULA at the receiver.
        d_h_tx (float): Inter-element spacing in the ULA at the transmitter.
        d_h_rx (float): Inter-element spacing in the ULA at the receiver.
        lambda_h (float): Carrier wavelength.

    Returns:
        np.ndarray: The LoS channel component.
    """
    # Compute the relative position vector and distance
    dx,dy = receiver_position - transmitter_position
    
    # ! potential error in the angle calculation for the aod
    angle_rad = np.arctan2(dy, dx)
    theta_h_AOD = angle_rad # Angle of Departure
    theta_h_AOA = (angle_rad + np.pi) % (2 * np.pi) # Angle of Arrival

    # Compute array responses for transmitter and receiver
    a_h_r = array_response(W_h_r, theta_h_AOA, d_h_rx, lambda_h)  # Receiver array response
    a_h_t = array_response(W_h_t, theta_h_AOD, d_h_tx, lambda_h)  # Transmitter array response

    # Compute the LoS channel component as outer product of array responses
    h_bar = a_h_r @ a_h_t.T # ? Array response provided in Bjornson & Demir (2024), pages 227 & 253; do not use the conjugate
    return h_bar


def computing_NLoS(W_h_t: int, W_h_r: int, numpy_generator : np.random._generator.Generator):
    """
    Computes the Non-Line-of-Sight (NLoS) component of the channel.

    Parameters:
        W_h_t (int): Number of elements in the ULA at the transmitter.
        W_h_r (int): Number of elements in the ULA at the receiver.
        numpy_generator (numpy.random._generator.Generator): generator used to draw new random values with numpy.

    Returns:
        np.ndarray: The NLoS channel component.
    """
    # Generate random complex Gaussian variables for the channel matrix
    # Real and imaginary parts are drawn from N(0, 1/2) to ensure variance of 1 for the complex numbers
    # ? Change the value of variance ? initially 1 in Peng 2022 but could be change to something else for proper NLOS modelisation
    h_nlos = (numpy_generator.standard_normal(size = (W_h_r, W_h_t) ) + 1j * numpy_generator.standard_normal( size = (W_h_r, W_h_t) ) ) / np.sqrt(2)
    return h_nlos


# ?  ----------------------------------------------     FUNCTIONS TO COMPUTE THE GAMMA TERMS FOR SINR CALCULATION    --------------------------------------------------------------


def Gamma_B_k(k, W, WWH, Theta_Phi, Phi_H_Theta_H,
              gains_transmitter_ris_receiver,
              H_BS_RIS, H_RIS_Users, kappa_S_d,
              H_Users_RIS, P_users, kappa_B_u_i, rho, sigma_k_squared):
    """Vectorized version of Gamma_B_k function."""
    K = W.shape[1]
    
    # Calculate all indices except k
    indices_except_k = np.arange(K)[np.arange(K) != k]
    
    # First term: Inter-user interference
    inter_user_interference_term = 0
    for indice in indices_except_k:
        #inter_user_interference_term += np.abs(H_RIS_Users[k].conj() @ Theta_Phi @ H_BS_RIS @ W[:, indice])**2
        pass
        inter_user_interference_term += np.abs(np.sqrt(gains_transmitter_ris_receiver[k]) * H_RIS_Users[k] @ Theta_Phi @ H_BS_RIS @ W[:, indice].reshape(-1, 1))**2

    # Third term: User-induced interference
    # For all users except k
    user_interference_not_k = 0
    for indice in indices_except_k:
        user_channel = np.sqrt(gains_transmitter_ris_receiver[k]) * H_Users_RIS[k].T @ Theta_Phi @ H_Users_RIS[indice]
        user_interference_not_k += (1 + kappa_B_u_i) * rho * P_users[indice] * np.abs(user_channel)**2
    
    # For user k
    user_k_channel = np.sqrt(gains_transmitter_ris_receiver[k]) * H_Users_RIS[k].conj().T @ Theta_Phi @ H_Users_RIS[k]
    user_interference_k = (1 + kappa_B_u_i) * P_users[k] * np.abs(user_k_channel)**2
    
    # Total user interference
    user_interference_term = user_interference_not_k  + user_interference_k 

    # Second term: Distortion noise caused by BS
    diag_matrix = np.diag(np.diag(WWH)).real
    distortion_term = 0
    #! Put to 0 because it disturbs a lot the learning. Need to be investigated to check if the issue is on the learning or the physics part.
    # TODO: Verify This term later on if we want to bring back the distortion term. Let's keep it simple for the moment
    #distortion_term = (kappa_S_d * H_RIS_Users[k] @ Theta_Phi @ H_BS_RIS @ diag_matrix @ H_BS_RIS.conj().T @ Phi_H_Theta_H @ H_RIS_Users[k].conj().T).real
    
    # Noise term
    sigma_noise_term = 1.1 * sigma_k_squared #TODO: what values for sigma_k_squared
    
    return np.squeeze(inter_user_interference_term + distortion_term + user_interference_term + sigma_noise_term)



def Gamma_S_k(K, k, Theta_Phi,
              gains_transmitter_ris_receiver,
              H_User_RIS, H_RIS_BS, f_u_k, P_users, kappa_B_u_i, delta_k_squared):
    """Computation of Gamma_S_k as per Peng 2022, Eq. (15)."""
    
    # Projected signal for all users through RIS and combining vector
    indices_except_k = np.arange(K)[np.arange(K) != k]

    # First term: interference from other users (i ≠ k)
    interference_other_user = 0
    for indice in indices_except_k:
        interference_other_user += np.squeeze(P_users[indice] * np.absolute(np.sqrt(gains_transmitter_ris_receiver[indice]) * f_u_k.T @ H_RIS_BS @ Theta_Phi @ H_User_RIS[indice]) ** 2) # inter user interference term
        # ? removing the .conj() in the first f_u_k above and in H_RIS_BS
    # Second term: distortion noise caused by all users
    # TODO add support for specific kappa_B_u_i for each user i. Currently same value is used for everyone.
    distortion_noise = kappa_B_u_i * ( interference_other_user + np.squeeze(P_users[k] * np.absolute(np.sqrt(gains_transmitter_ris_receiver[k]) * f_u_k.T @ H_RIS_BS @ Theta_Phi @ H_User_RIS[k]) ** 2) )
    # TODO: Verify This term later on if we want to bring back the distortion term. Let's keep it simple for the moment
    # ? removing the .conj() in the first f_u_k above and in H_RIS_BS

    # Third term: noise power scaled by combining vector norm
    third_term = np.linalg.norm(f_u_k)**2 * 1.1 * delta_k_squared #TODO: what values for delta_k_squared

    return interference_other_user + distortion_noise + third_term



def Gamma_E_d_k_l(k: int, l: int, W: np.ndarray, Theta_Phi: np.ndarray, Phi_H_Theta_H: np.ndarray,
                     diag_matrix_WWH: np.ndarray,
                     gains_transmitter_ris_receiver: np.ndarray,
                     H_BS_RIS: np.ndarray, H_RIS_Eaves_downlink: np.ndarray,
                     kappa_S_d: float, H_Users_RIS: np.ndarray, H_RIS_Eaves_uplink: np.ndarray,
                     P_users: np.ndarray, kappa_B_u_i: float,
                     mu_d_l_squared: float):
    """Vectorized version of Gamma_E_d_k_l function."""
    K = W.shape[1]
    
    # Calculate indices except k for third term
    indices = np.arange(K)
    indices_except_k = indices[indices != k]
    
    # Common computation for first and second terms (for all users)
    # H_RIS_Eaves_uplink[l].conj() @ Theta_Phi @ H_Users_RIS for all i
    common_matrix = np.sqrt(gains_transmitter_ris_receiver[K +l]) * H_RIS_Eaves_uplink[l] @ Theta_Phi @ H_Users_RIS  # Shape: [1, K]
    
    # Apply square root of P_users element-wise
    common_matrix_with_power = common_matrix * np.sqrt(P_users)
    
    # First term: sum of |common_matrix * sqrt(P_users)|^2 for all i
    first_term = np.sum(np.abs(common_matrix_with_power)**2)
    
    # Second term: kappa_B_u_i * first_term
    second_term = kappa_B_u_i * first_term
    
    # Third term: sum of |H_RIS_Eaves_downlink[l].conj() @ Theta_Phi @ H_BS_RIS @ w_i|^2 for i != k
    if indices_except_k.size > 0:  # Check if there are users other than k
        w_i_all = W[:, indices_except_k]  # All w_i except w_k
        interference_matrix = np.sqrt(gains_transmitter_ris_receiver[K +l]) * H_RIS_Eaves_downlink[l] @ Theta_Phi @ H_BS_RIS @ w_i_all
        third_term = np.sum(np.abs(interference_matrix)**2)
    else:
        third_term = 0
    
    # Fourth term: distortion noise caused by BS
    # TODO: Verify This term later on if we want to bring back the distortion term. Let's keep it simple for the moment
    fourth_term = (kappa_S_d * np.sqrt(gains_transmitter_ris_receiver[K +l]) * H_RIS_Eaves_downlink[l].conj() @ Theta_Phi @ H_BS_RIS @ diag_matrix_WWH @ H_BS_RIS.conj().T @ Phi_H_Theta_H @ H_RIS_Eaves_downlink[l].T * np.sqrt(gains_transmitter_ris_receiver[K +l])).real
    
    # Noise term
    mu_noise_term = 1.1 * mu_d_l_squared
    
    # Final result
    result = np.squeeze(first_term + second_term + third_term + fourth_term + mu_noise_term)
    
    return result


def Gamma_E_u_k_l(k, l, W, Theta_Phi, Phi_H_Theta_H,
                  diag_matrix_WWH,
                  gains_transmitter_ris_receiver,
                  H_BS_RIS, H_RIS_Eaves_downlink, kappa_S_d,
                  H_Users_RIS, H_RIS_Eaves_uplink, P_users, kappa_B_u_i):
    """Vectorized version of the Gamma_E_u_k_l function."""
    K = W.shape[1]
    
    # Calculate indices for first term (all except k)
    indices = np.arange(K)
    indices_except_k = indices[indices != k]
    
    # Common computation for all users
    # H_RIS_Eaves_uplink[l].conj() @ Theta_Phi @ H_Users_RIS for all i
    common_matrix = np.sqrt(gains_transmitter_ris_receiver[K+l]) * H_RIS_Eaves_uplink[l] @ Theta_Phi @ H_Users_RIS  # Shape: [1, K]
    
    # Apply square root of P_users element-wise
    common_matrix_with_power = common_matrix * np.sqrt(P_users)
    common_term_all = np.abs(common_matrix_with_power)**2
    
    # First term: sum of common_term for i != k
    first_term = np.sum(common_term_all[indices_except_k])
    
    # Second term: kappa_B_u_i * sum of common_term for all i
    second_term = kappa_B_u_i * np.sum(common_term_all)
    
    # Third term: sum of |H_RIS_Eaves_downlink[l].conj() @ Theta_Phi @ H_BS_RIS @ w_i|^2 for all i
    third_term_matrix = np.sqrt(gains_transmitter_ris_receiver[K+l]) * H_RIS_Eaves_downlink[l] @ Theta_Phi @ H_BS_RIS @ W
    third_term = np.sum(np.abs(third_term_matrix)**2)
    
    # Fourth term: distortion noise caused by BS
    # TODO: Verify This term later on if we want to bring back the distortion term. Let's keep it simple for the moment
    fourth_term = (kappa_S_d * np.sqrt(gains_transmitter_ris_receiver[K+l]) * H_RIS_Eaves_downlink[l].conj() @ Theta_Phi @ H_BS_RIS @ diag_matrix_WWH @ H_BS_RIS.conj().T @ Phi_H_Theta_H @ H_RIS_Eaves_downlink[l].T * np.sqrt(gains_transmitter_ris_receiver[K+l])).real
    
    # Final result
    result = np.squeeze(first_term + second_term + third_term + fourth_term)
    
    return result


# ?  ----------------------------------------------    UNIT CONVERSION FUNCTIONS ------------------------------------------------------------------------------------------------

def watts_to_db(watts):
    return 10 * np.log10(watts)

def watts_to_dbm(watts):
    dbm = 10 * np.log10(watts * 1000)
    return dbm

def dbm_to_watts(dbm):
    watts = (10 ** (dbm / 10)) / 1000
    return watts

def dBm_Hz_to_Watts(noise_power_density_dBm_per_Hz, channel_bandwidth_MHz):
    """
    Convert power spectral density (dBm/Hz) to total power in Watts over a given bandwidth.

    :param noise_power_density_dBm_per_Hz: Power Spectral Density in dBm/Hz
    :param channel_bandwidth_MHz: Bandwidth in MHz
    :return: Total power in Watts
    """
    # Convert dBm/Hz to Watts/Hz
    power_density_W_Hz = 10**(noise_power_density_dBm_per_Hz / 10) / 1000

    # Convert channel bandwidth from MHz to Hz
    channel_bandwidth_Hz = channel_bandwidth_MHz * 1e6 # Passing MHz in Hz

    # Compute total power in Watts
    total_power_W = power_density_W_Hz * channel_bandwidth_Hz

    return total_power_W

# ?  ----------------------------------------------     FUNCTIONS TO GENERATE THE UPLINK AND DOWNLINK SIGNALS     --------------------------------------------------------------


def generate_downlink_signal_BS(W, K, kappa_S_d, P_max, numpy_generator: np.random._generator.Generator):
    """
    Generate the signal transmitted by the BS (x_d) as defined in equation (5a),
    including the confidential signals s_d and distortion noise η_d.

    Args:
        W (np.ndarray): Beamforming matrix of shape (N_t, K), where N_t is the number of BS antennas, and K is the 
        number of users.
        K (int): number of users.
        kappa_S_d (float): HWI scale factor for BS's transmitter (η_d).
        P_max (float): Maximum transmit power allowed for the BS.
        numpy_generator (numpy.random._generator.Generator): generator used to draw new random values with numpy.

    Returns:
        x_d (np.ndarray): Transmitted signal from the BS to the users. Shape (N_t, 1).
        distortion_noise (np.ndarray): Distortion noise η_d of shape (N_t, 1).
        s_d (np.ndarray): Confidential information signals of shape (K, 1).
    """
    # Generate confidential signals s_d ~ CN(0, 1)
    s_d = ( numpy_generator.standard_normal( size = (K, 1) ) + 1j * numpy_generator.standard_normal( size = (K, 1) ) ) / np.sqrt(2)
    
    # Compute the precoded data signal x_d
    x_data = W @ s_d  # W is (N_t x K), s_d is (K x 1) => x_data is (N_t x 1)
    
    # Compute the distortion noise η_d
    WWH = W @ W.conj().T
    power_matrix = np.diag(WWH).real  # Diagonal of WWH (sum of power per antenna)
    # Compute the distortion noise η_d
    distortion_noise = np.sqrt(kappa_S_d) * (
        numpy_generator.standard_normal(size = x_data.shape) + 1j * numpy_generator.standard_normal( size = x_data.shape) ) / np.sqrt(2) * np.sqrt(power_matrix).reshape(-1, 1)  # Reshape to match (N_t, 1)
    # Compute the final BS signal x_d
    x_d = x_data + distortion_noise
    
    return x_d, distortion_noise, s_d



def generate_uplink_signal_users(P_k, K, kappa_B_u_k, numpy_generator: np.random._generator.Generator):
    """
    Generate the signals transmitted by all K users, x_{u,k}, as defined in equation (8b),
    including the independent Gaussian symbols s_{u,k}.

    Args:
        P_k (float): Transmit power of the k-th user.
        k (float): Number of users.
        kappa_B_u_k (float): HWI scale factor for the k-th user.
        numpy_generator (numpy.random._generator.Generator): generator used to draw new random values with numpy.

    Returns:
        x_u (np.ndarray): Transmitted signals from all users. Shape: (K, 1).
        distortion_noise (np.ndarray): Distortion noise η_{u,k} for all users. Shape: (K, 1).
        s_u (np.ndarray): Transmitted symbols by all users. Shape: (K, 1).
    """
    # Generate the users' information symbols s_{u,k} ~ CN(0, 1) for all K users
    s_u = (numpy_generator.standard_normal(size = (K, 1) ) + 1j * numpy_generator( size = (K, 1) ) ) / np.sqrt(2)  # Shape: (K, 1)
    
    # Scale the symbols to the users' transmit power
    s_tilde_u = np.sqrt(P_k).reshape(-1, 1) * s_u  # Element-wise scaling. Shape: (K, 1)
    
    # Generate distortion noise η_{u,k} ~ CN(0, kappa_B_u_k * |s_tilde_u_k|^2) for each user
    distortion_variance = kappa_B_u_k * np.abs(s_tilde_u)**2  # Shape: (K, 1)
    distortion_noise = np.sqrt(distortion_variance) * (
        ( numpy_generator.standard_normal( size = (K, 1) ) + 1j * numpy_generator.standard_normal(size = (K, 1) )) / np.sqrt(2)
    )  # Shape: (K, 1)
    
    # Compute the final signals for all users
    x_u = s_tilde_u + distortion_noise  # Shape: (K, 1)

    return x_u, distortion_noise, s_u