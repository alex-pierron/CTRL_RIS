import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import os
from copy import deepcopy
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from math import floor, log10

matplotlib.set_loglevel('warning')


def scaling_factor(power_patterns_array: np.ndarray, threshold: float = 1.5e1):
    def round_down_to_two_significant_digits(number):
        if number == 0:
            return 0
        exponent = floor(log10(abs(number)))  # Find the order of magnitude
        factor = 10 ** (exponent - 1)  # Compute the factor to extract first two digits
        significant_digits = floor(number / factor)  # Get first two significant digits
        return significant_digits * factor  # Reconstruct the number with rounded-down value

    means = np.mean(power_patterns_array, axis=1)
    # Replace zero means with a very small number to avoid division by zero
    means[means == 0] = np.finfo(float).eps  # Using the smallest possible float value
    
    scale_factors = threshold / means
    rounded_scale_factors = np.vectorize(round_down_to_two_significant_digits)(scale_factors)

    return rounded_scale_factors



def generate_eavesdropper_plots(user_positions, eavesdropper_positions, show=False, animation_duration=100, log_dir=None):
    """
    Generates plots for eavesdroppers' movements.

    Parameters:
    - user_positions: Positions of users.
    - eavesdropper_positions: Positions of eavesdroppers over iterations.
    - show: Whether to display the plots.
    - animation_duration: Duration of each frame in the animation.
    - log_dir: Directory to save the plots.

    Returns:
    - List of generated plotly figures.
    """

    # Validate input shapes
    if not isinstance(eavesdropper_positions, np.ndarray) or eavesdropper_positions.shape[2] != 2:
        raise ValueError("eavesdropper_positions must be a numpy array of shape (n_iter, n_eavesdroppers, 2)")

    if not isinstance(user_positions, np.ndarray) or user_positions.shape[1] != 2:
        raise ValueError("user_positions must be a numpy array of shape (n_users, 2)")

    # Extract number of iterations and eavesdroppers
    n_iter, n_eavesdroppers, _ = eavesdropper_positions.shape

    # Plot 1: Eavesdroppers' movement without trails
    fig1 = go.Figure(
        data=[
            go.Scatter(x=user_positions[:, 0], y=user_positions[:, 1],
                       mode='markers', marker=dict(size=10, color='blue'),
                       name='Users')
        ] + [
            go.Scatter(x=[eavesdropper_positions[0, i, 0]], y=[eavesdropper_positions[0, i, 1]],
                       mode='markers', marker=dict(size=10),
                       name=f'Eavesdropper {i+1}')
            for i in range(n_eavesdroppers)
        ],
        layout=go.Layout(
            title="Eavesdroppers' Movement (No Trails)",
            xaxis=dict(range=[0, 200]), yaxis=dict(range=[0, 200]),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, dict(frame=dict(duration=animation_duration, redraw=True), fromcurrent=True)])]
            )],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=user_positions[:, 0], y=user_positions[:, 1],
                               mode='markers', marker=dict(size=10, color='blue'),
                               name='Users')
                ] + [
                    go.Scatter(x=[eavesdropper_positions[k, i, 0]], y=[eavesdropper_positions[k, i, 1]],
                               mode='markers', marker=dict(size=10),
                               name=f'Eavesdropper {i+1}')
                    for i in range(n_eavesdroppers)
                ]
            ) for k in range(1, n_iter)
        ]
    )

    # Plot 2: Eavesdroppers' movement with trails
    fig2 = go.Figure(
        data=[
            go.Scatter(x=user_positions[:, 0], y=user_positions[:, 1],
                       mode='markers', marker=dict(size=10, color='blue'),
                       name='Users')
        ] + [
            go.Scatter(x=eavesdropper_positions[:, i, 0], y=eavesdropper_positions[:, i, 1],
                       mode='lines+markers', line=dict(dash='dash'),
                       marker=dict(size=6), name=f'Eavesdropper {i+1} Path')
            for i in range(n_eavesdroppers)
        ] + [
            go.Scatter(x=[eavesdropper_positions[-1, i, 0]], y=[eavesdropper_positions[-1, i, 1]],
                       mode='markers', marker=dict(size=12, symbol='circle'),
                       name=f'Eavesdropper {i+1} Last Position')
            for i in range(n_eavesdroppers)
        ],
        layout=go.Layout(
            title="Eavesdroppers' Movement with Trails",
            xaxis=dict(range=[0, 200]), yaxis=dict(range=[0, 200]),
            showlegend=True
        )
    )

    # Individual plots for each eavesdropper
    individual_figs = []
    for i in range(n_eavesdroppers):
        fig = go.Figure(
            data=[
                go.Scatter(x=user_positions[:, 0], y=user_positions[:, 1],
                           mode='markers', marker=dict(size=10, color='blue'),
                           name='Users'),

                go.Scatter(x=[eavesdropper_positions[0, i, 0]], y=[eavesdropper_positions[0, i, 1]],
                           mode='markers', marker=dict(size=12, color='orange', symbol='triangle-up'),
                           name=f'Eavesdropper {i+1} Start Position'),

                go.Scatter(x=eavesdropper_positions[:, i, 0], y=eavesdropper_positions[:, i, 1],
                           mode='lines+markers', line=dict(dash='dash'),
                           marker=dict(size=6), name=f'Eavesdropper {i+1} Path'),

                go.Scatter(x=[eavesdropper_positions[-1, i, 0]], y=[eavesdropper_positions[-1, i, 1]],
                           mode='markers', marker=dict(size=12, color='red', symbol='circle'),
                           name=f'Eavesdropper {i+1} Last Position')
            ],
            layout=go.Layout(
                title=f"Eavesdropper {i+1}'s Movement",
                xaxis=dict(range=[0, 200]), yaxis=dict(range=[0, 200]),
                showlegend=True
            )
        )
        individual_figs.append(fig)

    # Save plots if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

        fig1.write_html(os.path.join(log_dir, "eavesdroppers_no_trails.html"))
        fig2.write_html(os.path.join(log_dir, "eavesdroppers_with_trails.html"))

        for i, fig in enumerate(individual_figs):
            fig.write_html(os.path.join(log_dir, f"eavesdropper_{i+1}_movement.html"))

    # Show plots if specified
    if show:
        fig1.show()
        fig2.show()
        for fig in individual_figs:
            fig.show()

    return [fig1, fig2] + individual_figs





class SituationRenderer:
    """
    Class to render the situation of users and RIS.
    """

    def __init__(self, M,L, N_t,max_step_per_episode,
                 lambda_h, d_h, BS_position, RIS_position, users_position,
                 eavesdroppers_positions=None, eavesdropper_moving=False,
                 num_frames=200):
        """
        Initialize the SituationRenderer.

        Parameters:
        - M: Number of elements.
        - max_step_per_episode: Maximum steps per episode.
        - lambda_h: Wavelength.
        - d_h: Distance.
        - BS_position: Base station position.
        - RIS_position: RIS position.
        - users_position: Users' positions.
        - eavesdroppers_positions: Eavesdroppers' positions.
        - eavesdropper_moving: Whether eavesdroppers are moving.
        - num_frames: Number of frames for animation.
        """
        self.lambda_h = lambda_h
        self.M = M
        self.L = L
        self.N_t = N_t
        self.max_step_per_episode = max_step_per_episode
        self.d_h = d_h
        
        self.d_h_BS = d_h[0] * np.ones(N_t)  # ! Fix this to allow for a specific d_h for the BS
        self.BS_position = BS_position
        self.RIS_position = RIS_position
        self.users_position = np.array(users_position)
        self.eavesdropper_moving = eavesdropper_moving
        
        self.num_frames = num_frames
        
        self.current_frame = 0
        
        self.num_users = self.users_position.shape[0]

        if eavesdroppers_positions is not None:
            self.eavesdroppers_positions = np.array(eavesdroppers_positions)
            self.L = self.eavesdroppers_positions.shape[0]
        else:
            self.L = 0        
        self.eavesdropper_present = (self.L > 0)
        if self.eavesdropper_present:
            if self.eavesdropper_moving:
                self.eavesdroppers_positions = np.zeros((self.num_frames, 2))
            else:
                self.eavesdroppers_positions = np.array(eavesdroppers_positions)

        self.W_power_patterns = np.zeros((num_frames,self.num_users, 360))
        self.downlink_power_patterns = np.zeros((num_frames,self.num_users, 360))
        self.uplink_power_patterns = np.zeros((num_frames,self.num_users, 360))
        
        self.W_power_patterns = np.zeros((num_frames,self.num_users,360))
        self.rewards = np.zeros(self.num_frames)


        self.angles = np.linspace(0, 2 * np.pi, 360)



    def store_power_patterns(self, reward,
                             W_power_patterns,
                             Downlink_power_patterns,
                             Uplink_power_patterns):
        """
        Precompute data for all frames.

        Parameters:
        - Theta_Phi: Phase shifts.
        """
        self.rewards[self.current_frame] = reward
        self.W_power_patterns[self.current_frame] = deepcopy(W_power_patterns)
        self.downlink_power_patterns[self.current_frame] = deepcopy(Downlink_power_patterns)
        self.uplink_power_patterns[self.current_frame] = deepcopy(Uplink_power_patterns)
        self.current_frame += 1
        pass        


    def plotly_single_image(self,
                 W_power_patterns,
                 downlink_power_pattern,
                 uplink_power_pattern,
                 step_number,
                 reward,
                 save_dir,
                 name_file,
                 episode = None,
                 rectangle_coords=(30, 2.5, 40, 45)):
        """
        Generate a single image with subplots for power patterns and optionally overlay beam direction.

        Parameters:
        - phi_power_pattern: Phi power pattern data.
        - theta_power_pattern: Theta power pattern data.
        - theta_phi_power_pattern: Theta-Phi power pattern data.
        - W_power_patterns: Power patterns for W for each user.
        - step_number: Current step number for labeling.
        - reward: Current reward for labeling.
        - save_dir: Directory to save the image.
        - name_file: Name of the output file.
        - beam_direction: Optional beam direction data.
        """
        # Create subplots
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=["downlink_power_pattern", "uplink_power_pattern"])

        # Define power patterns
        RIS_power_patterns = [downlink_power_pattern, uplink_power_pattern]
        
        # Colors for different users
        W_colors = ['orange', 'navy', 'olive', 'cyan', "navy"]
        

        user_colors = ["purple", "blue","teal", "violet", "red", "teal", "magenta", "indigo"]
        # Extension factor for dashed lines
        extension_factor = 1.5  # Adjust for desired length

        # Compute direction vector for BS -> RIS
        bs_ris_vector = np.array(self.RIS_position) - np.array(self.BS_position)
        bs_ris_extended = np.array(self.BS_position) - extension_factor * bs_ris_vector  # Extend in the BS direction
        
        W_power_scale = scaling_factor(W_power_patterns, threshold=4e1)

        # Iterate through power patterns

        # Plot W Power Patterns for each user
        for j, W_power_pattern in enumerate(W_power_patterns):
            x_W = W_power_scale[j] * W_power_pattern * np.cos(self.angles) + self.BS_position[0]
            y_W = W_power_scale[j] * W_power_pattern * np.sin(self.angles) + self.BS_position[1]
            fig.add_trace(go.Scatter(
                x=x_W,
                y=y_W,
                mode='lines',
                line=dict(color=W_colors[j % len(W_colors)], width=2, dash='solid'),
                name=f"W Power Pattern for User {j+1}",
                legendgroup=f"W_Power_User{j+1}",
                showlegend=(True)
            ), row=1, col=1)

        for i, power_pattern in enumerate(RIS_power_patterns):

            # Add Base Station position (legend only in first subplot)
            fig.add_trace(go.Scatter(
                x=[self.BS_position[0]],
                y=[self.BS_position[1]],
                mode='markers',
                marker=dict(color='brown', size=10, symbol='square'),
                name="Base Station",
                legendgroup="BS",
                showlegend=(i == 0)
            ), row=1, col=i+1)

            # Add RIS position
            fig.add_trace(go.Scatter(
                x=[self.RIS_position[0]],
                y=[self.RIS_position[1]],
                mode='markers',
                marker=dict(color='green', size=10, symbol='diamond'),
                name="RIS",
                legendgroup="RIS",
                showlegend=(i == 0)
            ), row=1, col=i+1)

            # Scale factors
            power_scale = scaling_factor(power_pattern, threshold=1e1)
            for index_user, user_power_pattern in enumerate(power_pattern):
                # Plot RIS Power Pattern
                x_polar = power_scale[index_user] * user_power_pattern * np.cos(self.angles) + self.RIS_position[0]
                y_polar = power_scale[index_user] * user_power_pattern * np.sin(self.angles) + self.RIS_position[1]
                fig.add_trace(go.Scatter(
                    x=x_polar,
                    y=y_polar,
                    mode='lines',
                    line=dict(color=user_colors[index_user % len(user_colors)], width=2, dash='solid'),
                    name=f"RIS polar profile for user {index_user +1}",
                    legendgroup=f"user_{index_user +1}",
                    showlegend=(i == 0)
                ), row=1, col=i+1)

            # Draw dashed lines from RIS to Users
            for num_user, user_position in enumerate(self.users_position):
                ris_user_vector = np.array(user_position) - np.array(self.RIS_position)
                ris_user_extended = np.array(self.RIS_position) - extension_factor * ris_user_vector  # Extend towards RIS

                bs_user_vector = np.array(user_position) - np.array(self.BS_position)
                bs_user_extended = np.array(self.BS_position) - extension_factor * bs_user_vector  # Extend in the BS direction

                # Add Users positions
                fig.add_trace(go.Scatter(
                    x=[user_position[0]],
                    y=[user_position[1]],
                    mode='markers',
                    marker=dict(color=user_colors[num_user % len(user_colors)], size=8, symbol='circle'),
                    name=f"User {num_user +1}",
                    legendgroup=f"user_{num_user +1}",
                    showlegend=(i == 0)
                ), row=1, col=i+1)

                fig.add_trace(go.Scatter(
                    x=[ris_user_extended[0], user_position[0]],  
                    y=[ris_user_extended[1], user_position[1]],
                    mode='lines',
                    line=dict(color='gray', width=1.5, dash='dash'),
                    name="RIS to User (Extended)",
                    legendgroup="RIS_to_User",
                    showlegend=(i == 0)
                ), row=1, col=i+1)

                # Add the  BS → User line
                fig.add_trace(go.Scatter(
                    x=[bs_user_extended[0], user_position[0]],  
                    y=[bs_user_extended[1], user_position[1]],
                    mode='lines',
                    line=dict(color='gray', width=1.5, dash='dash'),
                    name="BS to User (Extended)",
                    legendgroup="BS_to_RIS",
                    showlegend=(i == 0)
                ), row=1, col=i+1)

            # Add the extended BS → RIS line
            fig.add_trace(go.Scatter(
                x=[bs_ris_extended[0], self.RIS_position[0]],  
                y=[bs_ris_extended[1], self.RIS_position[1]],
                mode='lines',
                line=dict(color='gray', width=1.5, dash='dash'),
                name="BS to RIS (Extended)",
                legendgroup="BS_to_RIS",
                showlegend=(i == 0)
            ), row=1, col=i+1)

            # Add Eavesdroppers and dotted lines if present and not moving
            if self.eavesdropper_present and not self.eavesdropper_moving:
                fig.add_trace(go.Scatter(
                    x=self.eavesdroppers_positions[:, 0],
                    y=self.eavesdroppers_positions[:, 1],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name="Eavesdroppers",
                    legendgroup="Eavesdroppers",
                    showlegend=(i == 0)
                ), row=1, col=i+1)

                for eavesdropper_position in self.eavesdroppers_positions:
                    fig.add_trace(go.Scatter(
                        x=[self.RIS_position[0], eavesdropper_position[0]],
                        y=[self.RIS_position[1], eavesdropper_position[1]],
                        mode='lines',
                        line=dict(color='red', width=1.5, dash='dot'),
                        name="RIS to Eavesdropper",
                        legendgroup="RIS_to_Eavesdropper",
                        showlegend=(i == 0)
                    ), row=1, col=i+1)


            # Update subplot titles and axis labels
            fig.update_xaxes(title_text="X (m)", row=1, col=i+1)
            fig.update_yaxes(title_text="Y (m)", row=1, col=i+1)
        
        # Add a single text annotation for step count and reward at the top center
        fig.add_annotation(
            x=0.5,  # Centered horizontally
            y=1.125,  # Slightly above the top but still visible
            xref="paper",
            yref="paper",
            text=f'<b>Episode: {episode} | Step: {step_number} | Reward: {reward:.4f}</b>',  # Bold for emphasis
            showarrow=False,
            font=dict(size=14),
            align="center"
        )

        # Add green rectangle if coordinates are provided
        if rectangle_coords is not None:
            x0, y0, x1, y1 = rectangle_coords
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="green", width=2),
                fillcolor="green",
                opacity=0.25,
                row='all', col='all'
            )
            
            # Add a custom legend entry for the rectangle
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='green', width=2),
                name="Obstacle",
                legendgroup="Obstacle",
                showlegend=True
            ))

        # Update layout to show only one legend
        fig.update_layout(title_text="Different Power Pattern", showlegend=True)

        # Ensure the directory exists
        output_dir = Path(save_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir))

        # Save the plot as an HTML file
        path_to_save = os.path.join(output_dir, name_file)
        fig.write_html(path_to_save)



    def render_situation(self, save_dir, name_file, plot_beam_arrows=False):
        """
        Render the situation and save as an animation, including beamforming vectors and W power patterns.

        Parameters:
        - save_dir: Directory to save the animation.
        - name_file: Name of the output file.
        - plot_beam_arrows: Boolean flag to optionally plot beam arrows.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(*self.BS_position, color='blue', marker='s', s=100, label="Base Station")
        ax.scatter(*self.RIS_position, color='green', marker='D', s=100, label="RIS")
        ax.scatter(self.users_position[:, 0], self.users_position[:, 1], color='black', marker='o', s=80, label="Users")

        if self.eavesdropper_present and not self.eavesdropper_moving:
            ax.scatter(self.eavesdroppers_positions[:, 0], self.eavesdroppers_positions[:, 1], color='red', marker='x', s=100, label="Eavesdroppers")

            
        RIS_scale_factor = 100
        x_polar = RIS_scale_factor * self.downlink_power_patterns[0] * np.cos(self.angles) + self.RIS_position[0]
        y_polar = RIS_scale_factor * self.downlink_power_patterns[0] * np.sin(self.angles) + self.RIS_position[1]
        polar_line, = ax.plot(x_polar, y_polar, color='purple', linestyle='-', linewidth=2, alpha=0.8, label="RIS polar profile")

        #step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        step_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Dynamic RIS Polar Profile with Varying Phases and Amplitudes")
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # Define a list of colors for different users
        colors = ['orange', 'brown', 'gray', 'olive', 'cyan']

        def update(frame):
            power_pattern = self.RIS_power_patterns[frame]
            x_polar = RIS_scale_factor * power_pattern * np.cos(self.angles) + self.RIS_position[0]
            y_polar = RIS_scale_factor * power_pattern * np.sin(self.angles) + self.RIS_position[1]
            polar_line.set_data(x_polar, y_polar)

            # Clear previous beamforming arrows and W power patterns
            for line in W_lines:
                line.remove()
            W_lines.clear()

            # Plot beamforming vectors for each user if plot_beam_arrows is True

            # Plot W power patterns for each user
            W_scale_factor = 1 # Dedicated scale factor for W power patterns
            W_power_patterns = self.W_power_patterns[frame]
            for i, W_power_pattern in enumerate(W_power_patterns):
                x_W = W_scale_factor * W_power_pattern * np.cos(self.angles) + self.BS_position[0]
                y_W = W_scale_factor * W_power_pattern * np.sin(self.angles) + self.BS_position[1]
                W_line, = ax.plot(x_W, y_W, color=colors[i % len(colors)], linestyle='-', linewidth=2, alpha=0.8, label=f"W Power Pattern for User {i+1}")
                W_lines.append(W_line)

            step_count = frame * (self.max_step_per_episode / self.num_frames)
            step_text.set_text(f'Step: {step_count} \n Reward: {round(self.rewards[frame],4)}')

            return polar_line, step_text
        W_lines = []

        anim = FuncAnimation(fig, update, frames=self.num_frames, interval=50, blit=True)

        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        output_dir = Path(save_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir))
        path_to_save = os.path.join(output_dir, name_file)
        anim.save(path_to_save, writer=writer)



    def plot_power_patterns(self,
                            num_frame):
        """
        Automate the plotting of power patterns for an undetermined number of users.

        Parameters:
        - power_patterns_users: List of power pattern arrays for each user.
        - power_patterns_bs: List of BS power pattern arrays for each user.
        - user_positions: List of user positions.
        - bs_position: Position of the base station.
        - RIS_position: Position of the RIS.
        - angles: Array of angles for power pattern computation.
        """
        colors = ['red', 'green', 'orange', 'blue', 'purple', 'brown']  # Extend as needed
        eavesdropper_color = 'gray'  # Color for eavesdroppers

        downlink_power_scale = scaling_factor(self.downlink_power_patterns[num_frame])
        uplink_power_scale = scaling_factor(self.uplink_power_patterns[num_frame])
        W_power_scale = scaling_factor(self.W_power_patterns[num_frame])

        # Create figure with subplots based on the number of users
        fig, axes = plt.subplots(1, self.num_users, figsize=(8 * self.num_users, 8))

        if self.num_users == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, ax in enumerate(axes):
            power_pattern_downlink_user = self.downlink_power_patterns[num_frame][i]
            power_pattern_uplink_user = self.uplink_power_patterns[num_frame][i]
            power_pattern_bs_user = self.W_power_patterns[num_frame][i]

            
            # Convert power pattern to Cartesian coordinates for plotting
            x_polar_downlink_user = downlink_power_scale[i] * power_pattern_downlink_user * np.cos(self.angles) + self.RIS_position[0]
            y_polar_downlink_user = downlink_power_scale[i] * power_pattern_downlink_user * np.sin(self.angles) + self.RIS_position[1]

            x_polar_uplink_user = uplink_power_scale[i] * power_pattern_uplink_user * np.cos(self.angles) + self.RIS_position[0]
            y_polar_uplink_user = uplink_power_scale[i] * power_pattern_uplink_user * np.sin(self.angles) + self.RIS_position[1]

            x_polar_bs_user =  W_power_scale[i] * power_pattern_bs_user * np.cos(self.angles) + self.BS_position[0]
            y_polar_bs_user = W_power_scale[i] * power_pattern_bs_user * np.sin(self.angles) + self.BS_position[1]

            # Plot user power pattern
            sc_downlink_user = ax.scatter(x_polar_downlink_user, y_polar_downlink_user, c=power_pattern_downlink_user, cmap='Blues', alpha=0.6, label=f'Downlink User {i+1} Power Pattern')
            
            ax.scatter(self.BS_position[0], self.BS_position[1], color='blue', edgecolors='black', marker='s', s=150, label='Base Station', zorder=3)
            
            ax.scatter(self.RIS_position[0], self.RIS_position[1], color='black', marker='D', s=120, label='RIS', zorder=3)
            
            sc_uplink_user = ax.scatter(x_polar_uplink_user, y_polar_uplink_user, c=power_pattern_uplink_user, cmap='Greens', alpha=0.6, label=f'Uplink User {i+1} Power Pattern')
            # Plot BS power pattern

            sc_bs_user = ax.scatter(x_polar_bs_user, y_polar_bs_user, c=power_pattern_bs_user, cmap='Reds', alpha=0.6, label=f'BS Power Pattern - User {i+1}')

            # Plot user positions
            for j, pos in enumerate(self.users_position):
                ax.scatter(pos[0], pos[1], color=colors[j % len(colors)], marker='o', edgecolors='black', s=140, label=f'User {j+1}', zorder=3)
                ax.text(pos[0] + 11, pos[1], f'User {j+1}', fontsize=12, color=colors[j % len(colors)], fontweight='bold')

            # Plot eavesdropper positions
            if self.eavesdropper_present and not self.eavesdropper_moving:
                for k, pos in enumerate(self.eavesdroppers_positions):
                    ax.scatter(pos[0], pos[1], color=eavesdropper_color, marker='x', s=140, label=f'Eavesdropper {k+1}', zorder=3)
                    ax.text(pos[0] + 9, pos[1], f'E {k+1}', fontsize=12, color=eavesdropper_color, fontweight='bold')
            
            ax.set_xlabel("X Position", fontsize=14, fontweight='bold')
            ax.set_ylabel("Y Position", fontsize=14, fontweight='bold')
            ax.set_title(f"Power Pattern - User {i+1}", fontsize=16, fontweight='bold', pad=20)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            ax.set_xlim(-205, 215)
            ax.set_ylim(-105, 215)

            # Add colorbars
            norm_down = Normalize(vmin=0, vmax=1)
            cbar_downlink_user = fig.colorbar(plt.cm.ScalarMappable(norm=norm_down, cmap='Blues'), ax=ax)
            cbar_downlink_user.set_label("Downlink Power Pattern Intensity (normalized)", fontsize=12)

            norm_up = Normalize(vmin=0, vmax=1)
            cbar_uplink_user = fig.colorbar(plt.cm.ScalarMappable(norm=norm_up, cmap='Greens'), ax=ax)
            cbar_uplink_user.set_label("Uplink Power Pattern Intensity (normalized)", fontsize=12)

            norm_bs = Normalize(vmin=0, vmax=1)
            cbar_bs_user = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bs, cmap='Reds'), ax=ax)
            cbar_bs_user.set_label("BS Power Pattern Intensity (normalized)", fontsize=12)

        plt.tight_layout()
        return fig


    def animate_power_patterns(self, power_patterns, save_dir, name_file):
        """
        Animate the best power patterns with support for multiple users.
        Each user gets their own row with two plots (downlink and uplink).
        All users are displayed on each subplot, but only the power patterns
        for the corresponding user are shown.
        
        Parameters:
        - power_patterns: Dictionary containing best power patterns.
        - save_dir: Directory to save the animation.
        - name_file: Name of the output file.
        """
        num_users = self.num_users
        
        # Create a figure with a grid: 2 columns (downlink/uplink) and num_users rows
        fig, axs = plt.subplots(num_users, 2, figsize=(16, 6 * num_users))
        column_titles = ["Downlink Power Pattern", "Uplink Power Pattern"]
        
        # Handle the case with only one user (axs might not be 2D)
        if num_users == 1:
            axs = np.array([axs])
        
        # User colors (similar to render_situation)
        user_colors = ["purple", "blue", "teal", "violet", "red", "magenta", "indigo"]
        W_colors = ['orange', 'navy', 'olive', 'cyan', "navy"]
    
        # Extension factor for dashed lines
        extension_factor = 1.5
        
        # Initialize lists to store line objects for animation
        downlink_lines = []
        uplink_lines = []
        W_lines_downlink = []
        #W_lines_uplink = []
        
        # Calculate the limits for all subplots based on all positions
        all_positions = np.vstack([np.array(self.BS_position), np.array(self.RIS_position)] + self.users_position)
        if self.eavesdropper_present and not self.eavesdropper_moving:
            all_positions = np.vstack([all_positions, self.eavesdroppers_positions])

        # Calculate the center of all points
        center_x = np.mean(all_positions[:,0])
        center_y = np.mean(all_positions[:,1])

        # Calculate the maximum distance from center to any point + margin for power patterns
        max_dist_x = np.max(np.abs(all_positions[:,0] - center_x))
        max_dist_y = np.max(np.abs(all_positions[:,1] - center_y))

        # Add margin for power patterns
        margin_factor = 1.5
        pattern_margin = max(max_dist_x, max_dist_y) * margin_factor

        # Define the limits
        """x_min = center_x - pattern_margin
        x_max = center_x + pattern_margin
        y_min = center_y - pattern_margin
        y_max = center_y + pattern_margin"""
        
        # Add column headers
        for col in range(2):
            fig.text(0.25 + 0.5 * col, 0.98, column_titles[col], 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Setup each subplot for each user
        for user_idx in range(num_users):
            row_axs = axs[user_idx]
            
            # Add row label for user
            fig.text(0.02, 0.5 - (user_idx - num_users/2 + 0.5) * (1.0/num_users), 
                    f"User {user_idx+1}", 
                    ha='left', va='center', fontsize=14, 
                    rotation=90, fontweight='bold',
                    color=user_colors[user_idx % len(user_colors)])
            
            for col_idx, ax in enumerate(row_axs):
                # Set title and labels
                ax.set_title(f"{column_titles[col_idx]} - User {user_idx+1}")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.grid(True)
                
                # Add Base Station
                ax.scatter(*self.BS_position, color='brown', marker='s', s=100, label="Base Station")
                
                # Add RIS
                ax.scatter(*self.RIS_position, color='green', marker='D', s=100, label="RIS")
                
                # Add ALL users with proper highlighting for the current user
                for i, user_position in enumerate(self.users_position):
                    if i == user_idx:
                        # Highlight the current user
                        ax.scatter(user_position[0], user_position[1], 
                                color=user_colors[i % len(user_colors)], 
                                marker='o', s=100, 
                                label=f"User {i+1} (Current)")
                    else:
                        # Display other users with smaller markers and more transparency
                        ax.scatter(user_position[0], user_position[1], 
                                color=user_colors[i % len(user_colors)], 
                                marker='o', s=60, alpha=0.5,
                                label=f"User {i+1}")
                

                # Add Eavesdroppers if present
                if self.eavesdropper_present and not self.eavesdropper_moving:
                    ax.scatter(self.eavesdroppers_positions[:, 0], self.eavesdroppers_positions[:, 1], 
                            color='red', marker='x', s=100, label="Eavesdroppers")
                    

                # Compute direction vector for BS -> RIS
                bs_ris_vector = np.array(self.RIS_position) - np.array(self.BS_position)
                bs_ris_extended = np.array(self.BS_position) - extension_factor * bs_ris_vector
                
                # Add BS to RIS line
                ax.plot([bs_ris_extended[0], self.RIS_position[0]], 
                        [bs_ris_extended[1], self.RIS_position[1]], 
                        'gray', linestyle='--', alpha=0.7, label="BS to RIS")
                
                # Add lines for RIS to Users and BS to Users
                for i, user_position in enumerate(self.users_position):
                    ris_user_vector = np.array(user_position) - np.array(self.RIS_position)
                    ris_user_extended = np.array(self.RIS_position) - extension_factor * ris_user_vector
                    
                    bs_user_vector = np.array(user_position) - np.array(self.BS_position)
                    bs_user_extended = np.array(self.BS_position) - extension_factor * bs_user_vector
                    
                    # Use different line styles and thickness based on if it's the current user
                    if i == user_idx:
                        # Current user - solid line with higher alpha
                        ax.plot([ris_user_extended[0], user_position[0]], 
                                [ris_user_extended[1], user_position[1]], 
                                'gray', linestyle='-', alpha=0.9, linewidth=1.5,
                                label="RIS to Current User" if i == 0 else "")
                        
                        ax.plot([bs_user_extended[0], user_position[0]], 
                                [bs_user_extended[1], user_position[1]], 
                                'gray', linestyle='-', alpha=0.9, linewidth=1.5,
                                label="BS to Current User" if i == 0 else "")
                    else:
                        # Other users - dashed lines with lower alpha
                        ax.plot([ris_user_extended[0], user_position[0]], 
                                [ris_user_extended[1], user_position[1]], 
                                'gray', linestyle='--', alpha=0.4,
                                label="RIS to Other User" if i == (user_idx+1) % num_users else "")
                        
                        ax.plot([bs_user_extended[0], user_position[0]], 
                                [bs_user_extended[1], user_position[1]], 
                                'gray', linestyle='--', alpha=0.4,
                                label="BS to Other User" if i == (user_idx+1) % num_users else "")
                        
                # Add lines for RIS to Eavesdroppers
                if self.eavesdropper_present and not self.eavesdropper_moving:
                    for i, eavesdropper_position in enumerate(self.eavesdroppers_positions):
                        if i == 0:  # Only label the first one
                            ax.plot([self.RIS_position[0], eavesdropper_position[0]], 
                                    [self.RIS_position[1], eavesdropper_position[1]], 
                                    'red', linestyle=':', alpha=0.7, label="RIS to Eavesdropper")
                        else:
                            ax.plot([self.RIS_position[0], eavesdropper_position[0]], 
                                    [self.RIS_position[1], eavesdropper_position[1]], 
                                    'red', linestyle=':', alpha=0.7)
                            
                # Set fixed limits for the subplot
                ax.set_xlim([-205,205])
                ax.set_ylim([-205,205])
                
                # Set equal aspect ratio for the subplot
                ax.set_aspect('equal')
            
            # Initialize power pattern lines for current user
            # Downlink power pattern
            dl_line, = row_axs[0].plot([], [], 
                                color=user_colors[user_idx % len(user_colors)], 
                                linestyle='-', linewidth=2, 
                                label=f"RIS polar profile")
            downlink_lines.append(dl_line)
            
            # Uplink power pattern
            ul_line, = row_axs[1].plot([], [], 
                                color=user_colors[user_idx % len(user_colors)], 
                                linestyle='-', linewidth=2, 
                                label=f"RIS polar profile")
            uplink_lines.append(ul_line)
            
            # W power pattern for downlink
            w_line_down, = row_axs[0].plot([], [], 
                                    color=W_colors[user_idx % len(W_colors)], 
                                    linestyle='-', linewidth=2, 
                                    label=f"W Power Pattern")
            W_lines_downlink.append(w_line_down)
            
            # W power pattern for uplink
            """w_line_up, = row_axs[1].plot([], [], 
                                    color=W_colors[user_idx % len(W_colors)], 
                                    linestyle='-', linewidth=2, 
                                    label=f"W Power Pattern")
            W_lines_uplink.append(w_line_up)"""
            
            # Add legend to each subplot with custom positioning to avoid overlap with data
            row_axs[0].legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, -0.1), ncol=3)
            row_axs[1].legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, -0.1), ncol=3)
        
        def update(frame):
            # Update title with step and reward information
            step = power_patterns["steps"][frame]
            reward = power_patterns["rewards"][frame]
            
            downlink_power_scale = scaling_factor(power_patterns["downlink_power_patterns"][frame])
            uplink_power_scale = scaling_factor(power_patterns["uplink_power_patterns"][frame])
            W_power_scale = scaling_factor(power_patterns["W_power_patterns"][frame], threshold=4e1)

            # Update power patterns for each user
            for user_idx in range(num_users):
                # Update W power pattern for the current user
                W_power_pattern = power_patterns["W_power_patterns"][frame][user_idx]
                x_W = W_power_scale[user_idx] * W_power_pattern * np.cos(self.angles) + self.BS_position[0]
                y_W = W_power_scale[user_idx] * W_power_pattern * np.sin(self.angles) + self.BS_position[1]
                W_lines_downlink[user_idx].set_data(x_W, y_W)
                #W_lines_uplink[user_idx].set_data(x_W, y_W)
                
                # Update downlink power pattern for the current user
                user_downlink_pattern = power_patterns["downlink_power_patterns"][frame][user_idx]
                x_polar_down = downlink_power_scale[user_idx] * user_downlink_pattern * np.cos(self.angles) + self.RIS_position[0]
                y_polar_down = downlink_power_scale[user_idx] * user_downlink_pattern * np.sin(self.angles) + self.RIS_position[1]
                downlink_lines[user_idx].set_data(x_polar_down, y_polar_down)
                
                # Update uplink power pattern for the current user
                user_uplink_pattern = power_patterns["uplink_power_patterns"][frame][user_idx]
                x_polar_up = uplink_power_scale[user_idx] * user_uplink_pattern * np.cos(self.angles) + self.RIS_position[0]
                y_polar_up = uplink_power_scale[user_idx] * user_uplink_pattern * np.sin(self.angles) + self.RIS_position[1]
                uplink_lines[user_idx].set_data(x_polar_up, y_polar_up)
            
            # Update the title with step and reward information
            fig.suptitle(f'Step: {step} | Reward: {reward:.4f}', fontsize=16, fontweight='bold')
            
            # Create a list of all artists that need to be updated
            artists = []
            artists.extend(downlink_lines)
            artists.extend(uplink_lines)
            artists.extend(W_lines_downlink)
            #artists.extend(W_lines_uplink)
            
            return artists
        
        # Adjust layout with more space for legends
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.98, hspace=0.5, wspace=0.2)
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(power_patterns["rewards"]), interval=50, blit=True)
        
        # Ensure the directory exists
        output_dir = Path(save_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir))
        
        # Save the animation
        path_to_save = os.path.join(output_dir, name_file)
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(path_to_save, writer=writer)
        
        plt.close(fig)





    def render_situation(self, save_dir, name_file, episode=None):
        """
        Render the situation and save as an animation, using a layout similar to animate_power_patterns.
        Each user gets their own row with two plots (downlink and uplink).

        Parameters:
        - save_dir: Directory to save the animation.
        - name_file: Name of the output file.
        - episode: Optional episode number to display in the title.
        """
        num_users = self.num_users
        
        # Create a figure with a grid: 2 columns (downlink/uplink) and num_users rows
        fig, axs = plt.subplots(num_users, 2, figsize=(16, 6 * num_users))
        column_titles = ["Downlink Power Pattern", "Uplink Power Pattern"]
        
        # Handle the case with only one user (axs might not be 2D)
        if num_users == 1:
            axs = np.array([axs])
        
        # User colors (similar to original)
        user_colors = ["purple", "blue", "teal", "violet", "red", "magenta", "indigo"]
        W_colors = ['orange', 'navy', 'olive', 'cyan', "navy"]
        
        # Extension factor for dashed lines
        extension_factor = 1.5
        
        # Add column headers
        for col in range(2):
            fig.text(0.25 + 0.5 * col, 0.98, column_titles[col], 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Initialize lists to store line objects for animation
        downlink_lines = []
        uplink_lines = []
        W_lines_downlink = []
        W_lines_uplink = []
        
        # Setup each subplot for each user
        for user_idx in range(num_users):
            row_axs = axs[user_idx]
            
            # Add row label for user
            fig.text(0.02, 0.5 - (user_idx - num_users/2 + 0.5) * (1.0/num_users), 
                    f"User {user_idx+1}", 
                    ha='left', va='center', fontsize=14, 
                    rotation=90, fontweight='bold',
                    color=user_colors[user_idx % len(user_colors)])
            
            for col_idx, ax in enumerate(row_axs):
                # Set title and labels
                ax.set_title(f"{column_titles[col_idx]} - User {user_idx+1}")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.grid(True)
                
                # Add Base Station
                ax.scatter(*self.BS_position, color='brown', marker='s', s=100, label="Base Station")
                
                # Add RIS
                ax.scatter(*self.RIS_position, color='green', marker='D', s=100, label="RIS")
                
                # Add ALL users with proper highlighting for the current user
                for i, user_position in enumerate(self.users_position):
                    if i == user_idx:
                        # Highlight the current user
                        ax.scatter(user_position[0], user_position[1], 
                                color=user_colors[i % len(user_colors)], 
                                marker='o', s=100, 
                                label=f"User {i+1} (Current)")
                    else:
                        # Display other users with smaller markers and more transparency
                        ax.scatter(user_position[0], user_position[1], 
                                color=user_colors[i % len(user_colors)], 
                                marker='o', s=60, alpha=0.5,
                                label=f"User {i+1}")
                
                # Add Eavesdroppers if present
                if self.eavesdropper_present and not self.eavesdropper_moving:
                    ax.scatter(self.eavesdroppers_positions[:, 0], self.eavesdroppers_positions[:, 1], 
                            color='red', marker='x', s=100, label="Eavesdroppers")
                    

                # Compute direction vector for BS -> RIS
                bs_ris_vector = np.array(self.RIS_position) - np.array(self.BS_position)
                bs_ris_extended = np.array(self.BS_position) - extension_factor * bs_ris_vector
                
                # Add BS to RIS line
                ax.plot([bs_ris_extended[0], self.RIS_position[0]], 
                        [bs_ris_extended[1], self.RIS_position[1]], 
                        'gray', linestyle='--', alpha=0.7, label="BS to RIS")
                
                # Add lines for RIS to Users and BS to Users
                for i, user_position in enumerate(self.users_position):
                    ris_user_vector = np.array(user_position) - np.array(self.RIS_position)
                    ris_user_extended = np.array(self.RIS_position) - extension_factor * ris_user_vector
                    
                    bs_user_vector = np.array(user_position) - np.array(self.BS_position)
                    bs_user_extended = np.array(self.BS_position) - extension_factor * bs_user_vector
                    
                    # Use different line styles and thickness based on if it's the current user
                    if i == user_idx:
                        # Current user - solid line with higher alpha
                        ax.plot([ris_user_extended[0], user_position[0]], 
                                [ris_user_extended[1], user_position[1]], 
                                'gray', linestyle='-', alpha=0.9, linewidth=1.5,
                                label="RIS to Current User" if i == 0 else "")
                        
                        ax.plot([bs_user_extended[0], user_position[0]], 
                                [bs_user_extended[1], user_position[1]], 
                                'gray', linestyle='-', alpha=0.9, linewidth=1.5,
                                label="BS to Current User" if i == 0 else "")
                    else:
                        # Other users - dashed lines with lower alpha
                        ax.plot([ris_user_extended[0], user_position[0]], 
                                [ris_user_extended[1], user_position[1]], 
                                'gray', linestyle='--', alpha=0.4,
                                label="RIS to Other User" if i == (user_idx+1) % num_users else "")
                        
                        ax.plot([bs_user_extended[0], user_position[0]], 
                                [bs_user_extended[1], user_position[1]], 
                                'gray', linestyle='--', alpha=0.4,
                                label="BS to Other User" if i == (user_idx+1) % num_users else "")
                
                # Add lines for RIS to Eavesdroppers
                if self.eavesdropper_present and not self.eavesdropper_moving:
                    for i, eavesdropper_position in enumerate(self.eavesdroppers_positions):
                        if i == 0:  # Only label the first one
                            ax.plot([self.RIS_position[0], eavesdropper_position[0]], 
                                    [self.RIS_position[1], eavesdropper_position[1]], 
                                    'red', linestyle=':', alpha=0.7, label="RIS to Eavesdropper")
                        else:
                            ax.plot([self.RIS_position[0], eavesdropper_position[0]], 
                                    [self.RIS_position[1], eavesdropper_position[1]], 
                                    'red', linestyle=':', alpha=0.7)
                            
                # Set fixed limits for consistent scaling
                ax.set_xlim([-205, 205])
                ax.set_ylim([-205, 205])
                
                # Set equal aspect ratio for the subplot
                ax.set_aspect('equal')
            
            # Initialize power pattern lines for current user
            # Downlink power pattern
            dl_line, = row_axs[0].plot([], [], 
                                color=user_colors[user_idx % len(user_colors)], 
                                linestyle='-', linewidth=2, 
                                label=f"RIS polar profile")
            downlink_lines.append(dl_line)
            
            # Uplink power pattern
            ul_line, = row_axs[1].plot([], [], 
                                color=user_colors[user_idx % len(user_colors)], 
                                linestyle='-', linewidth=2, 
                                label=f"RIS polar profile")
            uplink_lines.append(ul_line)
            
            # W power pattern for downlink
            w_line_down, = row_axs[0].plot([], [], 
                                    color=W_colors[user_idx % len(W_colors)], 
                                    linestyle='-', linewidth=2, 
                                    label=f"W Power Pattern")
            W_lines_downlink.append(w_line_down)
            
            # W power pattern for uplink
            """w_line_up, = row_axs[1].plot([], [], 
                                    color=W_colors[user_idx % len(W_colors)], 
                                    linestyle='-', linewidth=2, 
                                    label=f"W Power Pattern")
            W_lines_uplink.append(w_line_up)"""
            
            # Add legend to each subplot
            row_axs[0].legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, -0.1), ncol=3)
            row_axs[1].legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, -0.1), ncol=3)
        
        def update(frame):
            # Update title with step and reward information
            step_count = frame * (self.max_step_per_episode / self.num_frames)
            episode_text = f"Episode: {episode}" if episode is not None else "Episode: N/A"
            downlink_power_scale = scaling_factor(self.downlink_power_patterns[frame])
            uplink_power_scale = scaling_factor(self.uplink_power_patterns[frame])
            W_power_scale = scaling_factor(self.W_power_patterns[frame])
            # Update power patterns for each user
            for user_idx in range(num_users):
                # Update W power pattern for the current user
                W_power_pattern = self.W_power_patterns[frame][user_idx]
                x_W = W_power_scale[user_idx] * W_power_pattern * np.cos(self.angles) + self.BS_position[0]
                y_W = W_power_scale[user_idx] * W_power_pattern * np.sin(self.angles) + self.BS_position[1]
                W_lines_downlink[user_idx].set_data(x_W, y_W)
                #W_lines_uplink[user_idx].set_data(x_W, y_W)
                
                # Update downlink power pattern for the current user
                user_downlink_pattern = self.downlink_power_patterns[frame][user_idx]
                x_polar_down = downlink_power_scale[user_idx] * user_downlink_pattern * np.cos(self.angles) + self.RIS_position[0]
                y_polar_down = downlink_power_scale[user_idx] * user_downlink_pattern * np.sin(self.angles) + self.RIS_position[1]
                downlink_lines[user_idx].set_data(x_polar_down, y_polar_down)
                
                # Update uplink power pattern for the current user
                user_uplink_pattern = self.uplink_power_patterns[frame][user_idx]
                x_polar_up = uplink_power_scale[user_idx] * user_uplink_pattern * np.cos(self.angles) + self.RIS_position[0]
                y_polar_up = uplink_power_scale[user_idx] * user_uplink_pattern * np.sin(self.angles) + self.RIS_position[1]
                uplink_lines[user_idx].set_data(x_polar_up, y_polar_up)
            
            # Update the title with step and reward information
            fig.suptitle(f'{episode_text} | Step: {int(step_count)} | Reward: {self.rewards[frame]:.4f}', 
                        fontsize=16, fontweight='bold')
            
            # Create a list of all artists that need to be updated
            artists = []
            artists.extend(downlink_lines)
            artists.extend(uplink_lines)
            artists.extend(W_lines_downlink)
            
            return artists
        
        # Adjust layout with more space for legends
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.98, hspace=0.5, wspace=0.2)
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.num_frames, interval=50, blit=True)
        
        # Ensure the directory exists
        output_dir = Path(save_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir))
        
        # Save the animation
        path_to_save = os.path.join(output_dir, name_file)
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(path_to_save, writer=writer)
        
        plt.close(fig)