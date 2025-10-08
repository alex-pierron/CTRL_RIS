import numpy as np
from copy import deepcopy

class PositionGenerator:
    """
    A class used to generate random 2D positions for users within specified x and y limits.

    Attributes
    ----------
    grid_limits : np.ndarray
        A 2D numpy array containing the x and y limits for user positions, in the form [[x_min, x_max], [y_min, y_max]].
    
    Methods
    -------
    random_point_in_area(limits)
        Generates a random point within the area defined by the given limits.
    generate_random__users_positions(self.num_users, L)
        Generates random positions for self.num_users users within their respective areas.
    """

    def __init__(self,
                 num_users : int,
                 grid_limits,#? use a general grid limit or a specific one for eavesdropper and users ?
                 RIS_position,
                 numpy_generator, 
                 num_eavesdroppers : int = 0,
                 angle_difference_between_user = 10,
                 min_distance : float = 0.0):
        """
        Constructs all the necessary attributes for the PositionGenerator object.

        Parameters
        ----------
        grid_limits : np.ndarray
            A 2D numpy array defining the x and y limits for user positions, [[x_min, x_max], [y_min, y_max]].
        numpy_generator : np.random.Generator
            A numpy random number generator instance.
        """
        self.num_users = num_users
        self.grid_limits = grid_limits
        #self.eavesdropper_limits = eavesdropper_limits
        self.RIS_position = RIS_position
        self.numpy_generator = numpy_generator
        self.num_eavesdroppers = num_eavesdroppers
        self.angle_difference_between_user = angle_difference_between_user
        self.angle_is_max = False
        self.fully_random_positioning = False
        self.min_distance = min_distance

        self.min_angle = -np.arctan((self.RIS_position[1]-self.grid_limits[1][1])/(self.grid_limits[0][1]-self.RIS_position[0]))

        self.max_angle = -np.arctan((self.RIS_position[1]-self.grid_limits[1][0])/(self.grid_limits[0][0]-self.RIS_position[0]))
        
        self._intermediate_angle_toward_min = -np.arctan((self.RIS_position[1]-self.grid_limits[1,1])/(self.grid_limits[0,0]-self.RIS_position[0]))

        self._intermediate_angle_toward_max = -np.arctan((self.RIS_position[1]-self.grid_limits[1,0])/(self.grid_limits[0,1]-self.RIS_position[0]))

        self.min_distance_eavesdropper_users = 2
        self.max_distance_eavesdropper_users = np.inf

    def update_generation_condition(self,new_user_limits,
                                    is_angle_given_a_max, 
                                    new_angle_difference_between_user, 
                                    new_min_distance_between_eavesdropper_and_users,
                                    new_max_distance_between_eavesdropper_and_users,
                                    is_positioning_fully_random):
        """
        Updates the generation conditions for angle difference and distance.
        
        Parameters
        ----------
        new_user_limits : np.array
            2D limits in which the new positions can be sampled. Format is [ [x_min, x_max], [y_min, y_max] ].
        new_angle_difference_between_user : float
            New angle difference to have in radians when substracting the angles with the RIS for two users.
        new_min_distance_between_user : float
            New minimum distance between users.
        """
        self.grid_limits = new_user_limits
        self.angle_difference_between_user = new_angle_difference_between_user
        self.angle_is_max = is_angle_given_a_max
        self.fully_random_positioning = is_positioning_fully_random
        self.min_distance_eavesdropper_users = new_min_distance_between_eavesdropper_and_users
        self.max_distance_eavesdropper_users = new_max_distance_between_eavesdropper_and_users

        self.min_angle = -np.arctan((self.RIS_position[1]-self.grid_limits[1][1])/(self.grid_limits[0][1]-self.RIS_position[0]))

        self.max_angle = -np.arctan((self.RIS_position[1]-self.grid_limits[1][0])/(self.grid_limits[0][0]-self.RIS_position[0]))
        
        self._intermediate_angle_toward_min = -np.arctan((self.RIS_position[1]-self.grid_limits[1,1])/(self.grid_limits[0,0]-self.RIS_position[0]))

        self._intermediate_angle_toward_max = -np.arctan((self.RIS_position[1]-self.grid_limits[1,0])/(self.grid_limits[0,1]-self.RIS_position[0]))

        self.y_axis_upper_limit_aligned = (self.grid_limits[1][1] - self.RIS_position[1] ==0)

    def generate_new_users_positions(self):
        """
        Generate self.num_users user positions around RIS with angular constraints.
        Positions and angles are rounded to 0.1 precision.
        """
        # Determine sampling strategy

        if self.fully_random_positioning:
            return self.generate_random_users_positions()

        if self.num_users >= 3:
            zone_for_first_user_position_to_sample = self.numpy_generator.integers(low=-1, high=2, size=1)[0]
        else:
            zone_for_first_user_position_to_sample = 1
        
        user_coordinates = np.zeros((self.num_users, 2))
        
        if zone_for_first_user_position_to_sample == 1:
            # Anti-clockwise sampling from -π/2 to 0
            angles = self._generate_angles_anticlockwise()
        elif zone_for_first_user_position_to_sample == -1:
            # Clockwise sampling from -π/2 to 0
            angles = self._generate_angles_clockwise()
        else:  # zone == 0
            # Symmetric sampling around -π/2
            angles = self._generate_angles_symmetric()
        
        # Generate coordinates for all users with minimum distance constraint
        for i, angle in enumerate(angles):
            angle = np.round(angle, 2)  # Round angle to 0.01
            position = self._generate_position_at_angle(angle)
            
            # Ensure minimum distance from previously generated positions
            attempts = 0
            max_attempts = 1000
            while attempts < max_attempts:
                if i == 0 or self._check_minimum_distance(position, user_coordinates[:i]):
                    user_coordinates[i] = position
                    break
                else:
                    # Generate new position at the same angle
                    position = self._generate_position_at_angle(angle)
                    attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: use the last generated position
                user_coordinates[i] = position
        
        # Shuffle positions and return
        user_positions = self.numpy_generator.permutation(user_coordinates)
        return np.round(user_positions, 1)  # Round positions to 0.1


    def _generate_angles_anticlockwise(self):
        """Generate angles for anti-clockwise sampling strategy."""
        angles = []
        for i in range(self.num_users):
            if i == 0:
                # First user: sample near -π/2
                angle = self.numpy_generator.uniform(
                    low = self.max_angle,
                    high = self.max_angle + 2 * self.angle_difference_between_user
                )
            elif i == self.num_users - 1:
                # Last user: sample up to 0
                angle = self.numpy_generator.uniform(
                    low=angles[-1] + self.angle_difference_between_user,
                    high=max(angles[-1] + self.angle_difference_between_user, self.min_angle)
                )
            else:
                # Middle users: progressive sampling
                angle = self.numpy_generator.uniform(
                    low=angles[-1] + self.angle_difference_between_user,
                    high=angles[-1] + 2 * i * self.angle_difference_between_user
                )
            angles.append(angle)
        
        return angles

    def _generate_angles_clockwise(self):
        """Generate angles for clockwise sampling strategy."""
        angles = []
        
        for i in range(self.num_users):
            if i == 0:
                # First user: sample near 0
                angle = self.numpy_generator.uniform(
                    low=-2 * self.angle_difference_between_user,
                    high=self.min_angle
                )
            elif i == self.num_users - 1:
                # Last user: sample down to -π/2
                angle = self.numpy_generator.uniform(
                    low=self.max_angle,
                    high=min(self.max_angle, angles[-1] - self.angle_difference_between_user)
                )
            else:
                # Middle users: progressive sampling
                angle = self.numpy_generator.uniform(
                    low=-2 * i * self.angle_difference_between_user,
                    high=angles[-1] - self.angle_difference_between_user
                )
            
            angles.append(angle)
        
        return angles

    def _generate_angles_symmetric(self):
        """Generate angles for symmetric sampling around -π/2."""
        angles = []
        
        # Separate even and odd indices (excluding 0)
        even_indices = [i for i in range(self.num_users) if i != 0 and i % 2 == 0]
        odd_indices = [i for i in range(self.num_users) if i != 0 and i % 2 != 0]
        
        # Initialize tracking angles
        angle_central_user = self.numpy_generator.uniform(
            low=-np.pi/2 - self.angle_difference_between_user,
            high=-np.pi/2 + self.angle_difference_between_user
        )
        
        angle_even = deepcopy(angle_central_user)
        angle_odd = deepcopy(angle_central_user)
        
        for i in range(self.num_users):
            if i == 0:
                # Central user
                angles.append(angle_central_user)
            elif i in even_indices:
                # Even indices: sample towards 0
                if i == even_indices[-1]:
                    # Last even index
                    angle_even = self.numpy_generator.uniform(
                        low=angle_even + self.angle_difference_between_user,
                        high= self.min_angle
                    )
                else:
                    # Other even indices
                    angle_even = self.numpy_generator.uniform(
                        low=angle_even + self.angle_difference_between_user,
                        high=angle_even + 2 * self.angle_difference_between_user
                    )
                angles.append(angle_even)
            else:
                # Odd indices: sample towards -π/2
                if i == odd_indices[-1]:
                    # Last odd index
                    angle_odd = self.numpy_generator.uniform(
                        low=self.max_angle,
                        high=angle_odd - self.angle_difference_between_user
                    )
                else:
                    # Other odd indices
                    angle_odd = self.numpy_generator.uniform(
                        low=angle_odd - 2 * self.angle_difference_between_user,
                        high=angle_odd - self.angle_difference_between_user
                    )
                angles.append(angle_odd)
        
        return angles

    def _generate_position_at_angle(self, angle):
        """Generate a position at the specified angle within bounds."""
        
        #print(f"one angle is {angle},angle max is {self.max_angle}, angle intermediate is {self._intermediate_angle_toward_min, self._intermediate_angle_toward_max}")

        if self._intermediate_angle_toward_min > angle > self._intermediate_angle_toward_max:
            
            r_distance_min =  (self.grid_limits[0][0] - self.RIS_position[0] ) / np.cos(angle)
            r_distance_max = (self.grid_limits[0][1]- self.RIS_position[0] ) / np.cos(angle)
            
            sampled_distance = self.numpy_generator.uniform(low=r_distance_min, high=r_distance_max)

        elif angle <= self._intermediate_angle_toward_max:
            r_distance_min = (self.grid_limits[0][0] - self.RIS_position[0] ) / np.cos(angle) 

            r_distance_max = (self.RIS_position[1] - self.grid_limits[1, 0] ) / np.sin(-angle) 
            
        elif  angle >= self._intermediate_angle_toward_min and not self.y_axis_upper_limit_aligned:

            r_distance_min =  (self.grid_limits[1][1] - self.RIS_position[1] ) / np.sin(-angle)
            r_distance_max = (self.grid_limits[0][1]- self.RIS_position[0] ) / np.cos(angle)
            

        elif angle >= self._intermediate_angle_toward_min and self.y_axis_upper_limit_aligned:
            
            r_distance_min =  (self.grid_limits[0][0] - self.RIS_position[0] ) / np.cos(angle)
            r_distance_max = (self.grid_limits[0][1]- self.RIS_position[0] ) / np.cos(angle)
        
        sampled_distance = self.numpy_generator.uniform(low=r_distance_min, high=r_distance_max)

        position = np.array([
            self.RIS_position[0] + sampled_distance * np.cos(angle),
            self.RIS_position[1] + sampled_distance * np.sin(angle)
        ])
        
        return position
    
    
    def generate_new_eavesdroppers_positions(self, users_positions):
        """
        Generate positions for eavesdroppers based on user positions, ensuring that the distance 
        between each eavesdropper and all users is superior to self.min_distance_eavesdropper_users. 
        It also ensures that the distance between each eavesdropper and the closest user is inferior 
        to self.max_distance_eavesdropper_users. Finally the eavesdropper positions have to have 
        their x coordinate and y coordinate inside the lower and upper bounds given by self.grid_limits.
        Finally eavesdroppers positions are rounded to 0.1 precision.
        
        Parameters
        ----------
        users_positions : np.ndarray
            Array of shape (num_users, 2) containing user positions
        
        Returns
        -------
        np.ndarray
            Array of shape (self.num_eavesdroppers, 2) containing eavesdropper positions
        """
        # Pre-compute values
        x_min, x_max = self.grid_limits[0]
        y_min, y_max = self.grid_limits[1]
        min_dist_sq = self.min_distance_eavesdropper_users ** 2
        max_dist_sq = self.max_distance_eavesdropper_users ** 2
        
        eavesdropper_positions = np.empty((self.num_eavesdroppers, 2))
        
        for i in range(self.num_eavesdroppers):
            # Fast vectorized sampling with rejection sampling
            found = False
            attempts = 0
            max_attempts = 5000  # Reduced for faster fallback
            
            while not found and attempts < max_attempts:
                # Vectorized batch sampling for better performance
                batch_size = min(1000, max_attempts - attempts)
                candidates = np.column_stack([
                    self.numpy_generator.uniform(x_min, x_max, batch_size),
                    self.numpy_generator.uniform(y_min, y_max, batch_size)
                ])
                
                # Vectorized distance computation
                dist_sq = np.sum((candidates[:, np.newaxis] - users_positions) ** 2, axis=2)
                min_dist_sq_per_candidate = np.min(dist_sq, axis=1)
                
                # Vectorized constraint checking
                valid_mask = (
                    np.all(dist_sq >= min_dist_sq, axis=1) & 
                    (min_dist_sq_per_candidate <= max_dist_sq)
                )
                
                # Additional check for minimum distance between eavesdroppers
                if i > 0:
                    eavesdropper_dist_sq = np.sum((candidates[:, np.newaxis] - eavesdropper_positions[:i]) ** 2, axis=2)
                    min_eavesdropper_dist_sq = self.min_distance ** 2
                    eavesdropper_valid_mask = np.all(eavesdropper_dist_sq >= min_eavesdropper_dist_sq, axis=1)
                    valid_mask = valid_mask & eavesdropper_valid_mask
                
                if np.any(valid_mask):
                    # Take first valid candidate
                    valid_idx = np.argmax(valid_mask)
                    eavesdropper_positions[i] = candidates[valid_idx]
                    found = True
                
                attempts += batch_size
            
            if not found:
                # Fast fallback: sample around user positions
                eavesdropper_positions[i] = self._fast_fallback_position(users_positions, min_dist_sq, max_dist_sq)
        
        return np.round(eavesdropper_positions, 1)

    def _fast_fallback_position(self, users_positions, min_dist_sq, max_dist_sq):
        """Optimized fallback using geometric approach around users."""
        # Find user closest to grid center
        grid_center = np.array([(self.grid_limits[0][0] + self.grid_limits[0][1]) * 0.5,
                            (self.grid_limits[1][0] + self.grid_limits[1][1]) * 0.5])
        
        center_distances = np.sum((users_positions - grid_center) ** 2, axis=1)
        closest_user_idx = np.argmin(center_distances)
        closest_user = users_positions[closest_user_idx]
        
        # Try positions in a ring around the closest user
        min_dist = np.sqrt(min_dist_sq)
        max_dist = np.sqrt(max_dist_sq)
        
        # Use golden ratio for optimal sampling
        n_angles = 16
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        
        # Try at optimal distance (geometric mean of min and max)
        optimal_dist = np.sqrt(min_dist * max_dist) if max_dist < np.inf else min_dist * 1.5
        
        for angle in angles:
            candidate = closest_user + optimal_dist * np.array([np.cos(angle), np.sin(angle)])
            
            # Check bounds
            if (self.grid_limits[0][0] <= candidate[0] <= self.grid_limits[0][1] and 
                self.grid_limits[1][0] <= candidate[1] <= self.grid_limits[1][1]):
                
                # Quick distance check
                dist_sq = np.sum((candidate - users_positions) ** 2, axis=1)
                if np.all(dist_sq >= min_dist_sq) and np.min(dist_sq) <= max_dist_sq:
                    return candidate
        
        # Last resort: grid center
        return grid_center


    def random_point_in_area(self):
        """
        Generates a random point within the given x and y limits.

        Parameters
        ----------
        limits : np.ndarray
            A 2D numpy array defining the x and y limits, [[x_min, x_max], [y_min, y_max]].

        Returns
        -------
        tuple
            A tuple (x, y) representing the coordinates of the random point.
        """
        x_min, x_max = self.grid_limits[0]
        y_min, y_max = self.grid_limits[1]

        x = self.numpy_generator.uniform(x_min, x_max)
        y = self.numpy_generator.uniform(y_min, y_max)

        return round(x, 1), round(y, 1)

    def generate_random_users_positions(self):
        """
        Generates random positions for self.num_users users within their respective limits.

        Parameters
        ----------
        self.num_users : int
            The number of users for which to generate random positions.
        
        Returns
        -------
        tuple
            A tuple containing two lists:
            - users_position: A list of self.num_users tuples, each representing the (x, y) coordinates of a user's position.
        """
        def generate_unique_positions(count, existing_positions):
            positions = []
            attempts = 0
            max_attempts = 10000
            
            while len(positions) < count and attempts < max_attempts:
                position = self.random_point_in_area()
                position_array = np.array(position)
                
                # Check minimum distance from existing positions
                if len(positions) == 0 or self._check_minimum_distance(position_array, np.array(positions)):
                    positions.append(position)
                
                attempts += 1
            
            if len(positions) < count:
                # Fallback: generate remaining positions without distance constraint
                while len(positions) < count:
                    position = self.random_point_in_area()
                    if position not in existing_positions and position not in positions:
                        positions.append(position)
            
            return positions

        # Generate random positions for self.num_users users
        users_position = np.array(generate_unique_positions(self.num_users, set()))

        return users_position

    def generate_random_eavesdroppers_positions(self, existing_positions=None):
        """
        Generates random positions for self.num_eavesdroppers users within their respective limits.

        Parameters
        ----------
        existing_positions : np.ndarray, optional
            Array of shape (n, 2) containing existing positions to avoid.
        
        Returns
        -------
        np.ndarray
            Array of shape (self.num_eavesdroppers, 2) containing eavesdropper positions.
        """
        if existing_positions is None:
            existing_positions = np.empty((0, 2))
        
        def generate_unique_positions(count, existing_positions):
            positions = []
            attempts = 0
            max_attempts = 10000
            
            while len(positions) < count and attempts < max_attempts:
                position = self.random_point_in_area()
                position_array = np.array(position)
                
                # Check minimum distance from existing positions and previously generated positions
                all_existing = np.vstack([existing_positions, np.array(positions)]) if len(positions) > 0 else existing_positions
                
                if len(all_existing) == 0 or self._check_minimum_distance(position_array, all_existing):
                    positions.append(position)
                
                attempts += 1
            
            if len(positions) < count:
                # Fallback: generate remaining positions without distance constraint
                while len(positions) < count:
                    position = self.random_point_in_area()
                    if position not in [tuple(p) for p in existing_positions] and position not in positions:
                        positions.append(position)
            
            return positions

        # Generate random positions for self.num_eavesdroppers eavesdroppers
        eavesdroppers_positions = np.array(generate_unique_positions(self.num_eavesdroppers, existing_positions))

        return eavesdroppers_positions

    def _check_minimum_distance(self, position, existing_positions):
        """
        Check if a position maintains minimum distance from existing positions.
        
        Parameters
        ----------
        position : np.ndarray
            Position to check (shape: (2,))
        existing_positions : np.ndarray
            Array of existing positions (shape: (n, 2))
        
        Returns
        -------
        bool
            True if position maintains minimum distance, False otherwise
        """
        if len(existing_positions) == 0:
            return True
        
        distances = np.sqrt(np.sum((existing_positions - position) ** 2, axis=1))
        return np.all(distances >= self.min_distance)