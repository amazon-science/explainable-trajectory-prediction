# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model Components classes.

    These classes are derived from the thirdparty/trajectron/model
    which are part of the Trajectron++ repository at:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import model.components as thirdparty_components
import model.dynamics as thirdparty_dynamics
import torch
import utils


def get_min_max(data):
    return (
        data[..., 0].min().item(),
        data[..., 0].max().item(),
        data[..., 1].min().item(),
        data[..., 1].max().item(),
    )


class GMM2D(thirdparty_components.GMM2D):
    """The predicted Gaussian mixture model over the future trajectories."""

    def __init__(self, log_weights, mus, log_sigmas, corrs):
        super().__init__(log_weights, mus, log_sigmas, corrs)

    def mode(self, resolution):
        """Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid.

        Args:
            resolution: The grid resolution.

        Return:
            The mode of the Gaussian mixture model.
        """
        if self.mus.shape[-2] > 1:
            number_of_samples, number_of_node, number_of_timesteps, _, _ = self.mus.shape
            if number_of_samples != 1:
                raise ValueError("For taking the mode only one sample makes sense")
            node_maximum_likelihood_modes = []
            for node_index in range(number_of_node):
                timestep_maximum_likelihood_modes = []
                for timestep in range(number_of_timesteps):
                    gmm = self.get_for_node_at_time(node_index, timestep)
                    x_min, x_max, y_min, y_max = get_min_max(self.mus[:, node_index, timestep, :])
                    search_grid = (
                        torch.stack(
                            torch.meshgrid(
                                [
                                    torch.arange(x_min, x_max, resolution),
                                    torch.arange(y_min, y_max, resolution),
                                ]
                            ),
                            dim=2,
                        )
                        .view(-1, 2)
                        .float()
                        .to(self.device)
                    )
                    ll_score = gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    timestep_maximum_likelihood_modes.append(search_grid[argmax])
                node_maximum_likelihood_modes.append(
                    torch.stack(timestep_maximum_likelihood_modes, dim=0)
                )
            return torch.stack(node_maximum_likelihood_modes, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def get_for_node(self, node_index):
        """Return the GMM object for a single node."""
        return self.__class__(
            self.log_pis[:, node_index : node_index + 1],
            self.mus[:, node_index : node_index + 1],
            self.log_sigmas[:, node_index : node_index + 1],
            self.corrs[:, node_index : node_index + 1],
        )


class SingleIntegrator(thirdparty_dynamics.SingleIntegrator):
    def integrate_distribution(self, velocity_dist, input_features=None):
        """Integrates the GMM velocity distribution to a distribution over position.

        This function is almost the same as th one from the thirdparty except
        it returns the overrided GMM2D object

            Args:
                velocity_dist: Joint GMM Distribution over velocity in x and y direction.
                input_features: The feature map of all encoded input features.

            Return:
                Joint GMM Distribution over position in x and y direction.
        """
        prediction_horizon = velocity_dist.mus.shape[-3]
        sample_batch_dim = list(velocity_dist.mus.shape[0:2])

        position_dist_sigma_matrix_list = []
        velocity_dist_sigma_matrix = velocity_dist.get_covariance_matrix()
        position_dist_sigma_matrix_t = torch.zeros(
            sample_batch_dim + [velocity_dist.components, 2, 2], device=self.device
        )
        for timestep in range(prediction_horizon):
            velocity_sigma_matrix_t = velocity_dist_sigma_matrix[:, :, timestep]
            full_sigma_matrix_t = utils.block_diag(
                [position_dist_sigma_matrix_t, velocity_sigma_matrix_t]
            )
            position_dist_sigma_matrix_t = self.F[..., :2, :].matmul(
                full_sigma_matrix_t.matmul(self.F_t)[..., :2]
            )
            position_dist_sigma_matrix_list.append(position_dist_sigma_matrix_t)

        position_dist_sigma_matrix = torch.stack(position_dist_sigma_matrix_list, dim=2)
        initial_position = self.initial_conditions["pos"].unsqueeze(1)
        if initial_position.size()[0] != input_features.size()[0]:
            initial_position = initial_position.repeat(input_features.size()[0], 1, 1)

        position_mus = initial_position[:, None] + torch.cumsum(velocity_dist.mus, dim=2) * self.dt
        return GMM2D.from_log_pis_mus_cov_mats(
            velocity_dist.log_pis, position_mus, position_dist_sigma_matrix
        )


class Unicycle(thirdparty_dynamics.Unicycle):
    def integrate_distribution(self, control_dist, input_features):
        """Integrates the GMM control distribution to a distribution over position.

        This function is almost the same as th one from the thirdparty except
        it returns the overrided GMM2D object

            Args:
                control_dist: Joint GMM Distribution over control commands.
                input_features: The feature map of all encoded input features.

            Return:
                Joint GMM Distribution over position in x and y direction.
        """
        sample_batch_dim = list(control_dist.mus.shape[0:2])
        prediction_horizon = control_dist.mus.shape[-3]
        position_present = self.initial_conditions["pos"].unsqueeze(1)
        velocity_present = self.initial_conditions["vel"].unsqueeze(1)

        if position_present.size()[0] != input_features.size()[0]:
            position_present = position_present.repeat(input_features.size()[0], 1, 1)
            velocity_present = velocity_present.repeat(input_features.size()[0], 1, 1)

        degree_present = torch.atan2(velocity_present[..., 1], velocity_present[..., 0])
        degree_present = degree_present + torch.tanh(
            self.p0_model(torch.cat((input_features, degree_present), dim=-1))
        )

        dist_sigma_matrix = control_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(
            sample_batch_dim + [control_dist.components, 4, 4], device=self.device
        )
        u = torch.stack([control_dist.mus[..., 0], control_dist.mus[..., 1]], dim=0)
        x = torch.stack(
            [
                position_present[..., 0],
                position_present[..., 1],
                degree_present,
                torch.norm(velocity_present, dim=-1),
            ],
            dim=0,
        )
        pos_dist_sigma_matrix_list = []
        mus_list = []
        for t in range(prediction_horizon):
            F_t = self.compute_jacobian(sample_batch_dim, control_dist.components, x, u[:, :, :, t])
            G_t = self.compute_control_jacobian(
                sample_batch_dim, control_dist.components, x, u[:, :, :, t]
            )
            dist_sigma_matrix_t = dist_sigma_matrix[:, :, t]
            pos_dist_sigma_matrix_t = F_t.matmul(
                pos_dist_sigma_matrix_t.matmul(F_t.transpose(-2, -1))
            ) + G_t.matmul(dist_sigma_matrix_t.matmul(G_t.transpose(-2, -1)))
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t[..., :2, :2])

            x = self.dynamic(x, u[:, :, :, t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        pos_mus = torch.stack(mus_list, dim=2)
        return GMM2D.from_log_pis_mus_cov_mats(control_dist.log_pis, pos_mus, pos_dist_sigma_matrix)


class AdditiveAttention(thirdparty_components.AdditiveAttention):
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super().__init__(encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim)

    def forward(self, encoder_states, decoder_state, mask=None):
        """Compute the attention scores of edges and apply them.

        Args:
            encoder_states: The features encoding the edges to neighbors
            decoder_state: The features encoding the history of the target agent.
            mask: A multi-values mask which represents the status of the neighbor.

        Return:
            The weighted averaged edges encoding.
            The attention scores.
        """
        score_vec = torch.cat(
            [
                self.score(encoder_states[:, i], decoder_state)
                for i in range(encoder_states.shape[1])
            ],
            dim=1,
        )
        if mask is not None and mask.shape[-1] == score_vec.shape[-1]:
            score_vec[mask == 0] = -float("inf")  # for padded neighbors
            score_vec[mask == 2] = -float("inf")  # for random neighbors
        attention_probs = torch.unsqueeze(torch.nn.functional.softmax(score_vec, dim=1), dim=2)
        attention_probs = torch.where(
            torch.isnan(attention_probs), torch.zeros_like(attention_probs), attention_probs
        )
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        return final_context_vec, attention_probs
