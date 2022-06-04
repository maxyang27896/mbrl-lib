# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gym
import copy
import mbrl.models.util as model_util
import numpy as np
import torch

import mbrl.types

from . import Model

from mbrl.util.math import euler_to_quaternion, quaternion_rotation_matrix

class ModelEnvPushing:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = self.termination if (termination_fn==None) else termination_fn
        self.reward_fn = self.reward if (reward_fn==None) else reward_fn
        self.device = model.device

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.observation_mode = env.observation_mode
        self.termination_pos_dist = env.termination_pos_dist
        self.terminate_early = env.terminate_early
        self.traj_n_points = env.traj_n_points
        self.TCP_lims = env.robot.arm.TCP_lims

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True
        

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state

    def reset_batch_goals(
        self, 
        batch_size: int
        ):
        '''
        Set the goal batches and target_ids to match with current goals of the gym
        environment.
        '''
        self.targ_traj_list_id = torch.tensor(self.env.targ_traj_list_id).to(self.device)
        # TODO
        if self.targ_traj_list_id>=1:
            print(self.env.targ_traj_list_id)
        if self.targ_traj_list_id < 0:
            print("Error targ_traj_list_id is below 0, the gym environment has not been reset.")
        
        self.traj_pos_workframe = model_util.to_tensor(copy.deepcopy(self.env.traj_pos_workframe)).to(torch.float32).to(self.device)
        self.traj_rpy_workframe = model_util.to_tensor(copy.deepcopy(self.env.traj_rpy_workframe)).to(torch.float32).to(self.device)
        self.traj_orn_workframe = model_util.to_tensor(copy.deepcopy(self.env.traj_orn_workframe)).to(torch.float32).to(self.device)

        self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
        self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id]
        self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id]

        # Create the batches needed to evaluate actions
        self.targ_traj_list_id_batch = torch.tile(self.targ_traj_list_id, (batch_size,))
        self.goal_pos_workframe_batch = torch.tile(self.goal_pos_workframe, 
                                        (batch_size,) 
                                        + tuple([1] * self.goal_pos_workframe.ndim))
        self.goal_orn_workframe_batch = torch.tile(self.goal_orn_workframe, 
                                        (batch_size,) 
                                        + tuple([1] * self.goal_orn_workframe.ndim))
        
    def outside_tcp_lims(self, tcp_pos_workframe, cur_obj_pos_workframe):
        '''
        Function to check if either object or tcp postion is outside of the 
        TCP limit
        '''
        return ((tcp_pos_workframe[:, 0] < self.TCP_lims[0,0]) | 
            (tcp_pos_workframe[:, 0] > self.TCP_lims[0,1]) | 
            (tcp_pos_workframe[:, 1] < self.TCP_lims[1,0]) | 
            (tcp_pos_workframe[:, 1] > self.TCP_lims[1,1]) | 
            (cur_obj_pos_workframe[:, 0] < self.TCP_lims[0,0]) | 
            (cur_obj_pos_workframe[:, 0] > self.TCP_lims[0,1]) | 
            (cur_obj_pos_workframe[:, 1] < self.TCP_lims[1,0]) | 
            (cur_obj_pos_workframe[:, 1] > self.TCP_lims[1,1]))

    def termination(
        self, 
        act: torch.Tensor, 
        next_obs: torch.Tensor
        ) -> torch.Tensor:
        """
        Criteria for terminating an episode. Should return a vector of dones of size 
        population_size x batch_size
        """

        batch_size = next_obs.shape[0]
        
        # For different observation modes
        if 'reduced' in self.observation_mode: 
            tcp_pos_to_goal_workframe = next_obs[:, 0:3]
            # tcp_orn_to_goal_workframe = next_obs[:, 3:7]
            # tcp_lin_vel_workframe = next_obs[:, 7:10]
            # tcp_ang_vel_workframe = next_obs[:, 10:13]
            cur_obj_pos_to_goal_workframe = next_obs[:, 13:16]
            # cur_obj_orn_to_goal_workframe = next_obs[:, 16:20]
            # cur_obj_lin_vel_workframe = next_obs[:, 20:23]
            # cur_obj_ang_vel_workframe = next_obs[:, 23:26]

            tcp_pos_workframe = tcp_pos_to_goal_workframe + self.goal_pos_workframe_batch
            cur_obj_pos_workframe = cur_obj_pos_to_goal_workframe + self.goal_pos_workframe_batch

            # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_workframe - self.goal_pos_workframe_batch, axis=1)

        elif 'tactile_pose' in self.observation_mode: 
            tcp_pos_to_goal_workframe = next_obs[:, 0:3]
            # tcp_orn_to_goal_workframe = next_obs[:, 3:7]
            cur_obj_pos_to_goal_workframe = next_obs[:, 7:10]
            # cur_obj_orn_to_goal_workframe = next_obs[:, 10:14]

            tcp_pos_workframe = tcp_pos_to_goal_workframe[:, 0:2] + self.goal_pos_workframe_batch[:, 0:2]
            cur_obj_pos_workframe = cur_obj_pos_to_goal_workframe[:, 0:2] + self.goal_pos_workframe_batch[:, 0:2]

             # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_to_goal_workframe, axis=1)
        # Default oracle observations
        else:
            tcp_pos_workframe = next_obs[:, 0:3]
            # tcp_rpy_workframe = next_obs[:, 3:6]
            # tcp_lin_vel_workframe = next_obs[:, 6:9]
            # tcp_ang_vel_workframe = next_obs[:, 9:12]
            cur_obj_pos_workframe = next_obs[:, 12:15]
            # cur_obj_rpy_workframe = next_obs[:, 15:18]
            # cur_obj_lin_vel_workframe = next_obs[:, 18:21]
            # cur_obj_ang_vel_workframe = next_obs[:, 21:24]
            # pred_goal_pos_workframe = next_obs[:, 24:27]

            # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_workframe - self.goal_pos_workframe_batch, axis=1)

        # Update goals index if for those that subgoals reached
        self.targ_traj_list_id_batch[obj_goal_pos_dist < self.termination_pos_dist] += 1

        # Create terminated vector, terminated is true if last subgoal is reached
        terminated = torch.zeros((batch_size, 1), dtype=bool).to(self.device)
        terminated[self.targ_traj_list_id_batch >= self.traj_n_points] = True

        # Early termination if outside of the tcp limits
        if self.terminate_early:
            terminated[self.outside_tcp_lims(tcp_pos_workframe, cur_obj_pos_workframe)] = True
        
        # Update goal position batch for none terminated samples
        self.goal_pos_workframe_batch[~terminated[:,0]] = self.traj_pos_workframe[self.targ_traj_list_id_batch[~terminated[:,0]]]

        return terminated

    def xyz_obj_dist_to_goal(
        self, 
        obj_pos_data: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate distance object position to goal position. 
        """
        # obj to goal distance
        return torch.linalg.norm(obj_pos_data - self.goal_pos_workframe_batch, axis=1)

    def get_pos_dist(
        self, 
        obj_pos_data: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate distance between two vector using a distance vector. 
        """
        # calculate distance
        return torch.linalg.norm(obj_pos_data, axis=1)

    def orn_obj_dist_to_goal(
        self, 
        obj_orn_data: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate distance (angle) between the current obj orientation and goal orientation. 
        """

        # obj to goal orientation
        cur_obj_orn_workframe = euler_to_quaternion(obj_orn_data)
        inner_product = torch.sum(self.goal_orn_workframe_batch*cur_obj_orn_workframe, 1)
        return torch.arccos(torch.clip(2 * (inner_product ** 2) - 1, -1, 1))

    def get_orn_dist(
        self, 
        obj_orn_data: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate the distance (angle) between two quarternion using a orientation 
        difference quaternion.
        """
        # obj to goal orientation
        return torch.arccos(torch.clip(
            (2 * (obj_orn_data[:, 3]**2)) - 1, -1, 1))

    def cos_tcp_dist_to_obj(
        self, 
        obj_orn_data_workframe: torch.Tensor, 
        tcp_orn_data_workframe: torch.Tensor
        ) -> torch.Tensor:
        """
        Cos distance from current orientation of the TCP to the current
        orientation of the object
        """
        
        batch_size = obj_orn_data_workframe.shape[0]

        # tip normal to object normal
        # In reduced the orientation data is querternion
        if 'reduced' or 'tactile_pose' in self.observation_mode: 
            obj_rot_matrix_workframe = quaternion_rotation_matrix(obj_orn_data_workframe)
            tip_rot_matrix_workframe = quaternion_rotation_matrix(tcp_orn_data_workframe)
        # In other obersavation mode the orientation data is euler
        else:
            cur_obj_orn_workframe = euler_to_quaternion(obj_orn_data_workframe)
            obj_rot_matrix_workframe = quaternion_rotation_matrix(cur_obj_orn_workframe)
            tcp_orn_workframe = euler_to_quaternion(tcp_orn_data_workframe)
            tip_rot_matrix_workframe = quaternion_rotation_matrix(tcp_orn_workframe)

        obj_rot_matrix_workframe = torch.reshape(obj_rot_matrix_workframe, (batch_size, 3, 3))
        obj_init_vector_workframe = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(self.device)
        obj_vector_workframe = torch.matmul(obj_rot_matrix_workframe, obj_init_vector_workframe)
        # obj_vector_workframe = obj_rot_matrix_workframe[:, :, 0]

        tip_rot_matrix_workframe  = torch.reshape(tip_rot_matrix_workframe, (batch_size, 3, 3))
        tip_init_vector_workframe  = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(self.device)
        tip_vector_workframe  = torch.matmul(tip_rot_matrix_workframe, tip_init_vector_workframe)
        # tip_vector_workframe = tip_rot_matrix_workframe[:, :, 0]

        obj_tip_dot_product = torch.sum(obj_vector_workframe*tip_vector_workframe, 1)
        cos_sim_workfrfame = obj_tip_dot_product / (
            torch.linalg.norm(obj_vector_workframe, axis=1) * torch.linalg.norm(tip_vector_workframe, axis=1)
        )
        cos_dist_workframe = 1 - cos_sim_workfrfame

        return cos_dist_workframe

    def cos_tcp_Rz_dist_to_obj(
        self,
        cos_obj_Rz_to_goal_workframe: torch.Tensor,  
        cos_tcp_Rz_to_goal_workframe: torch.Tensor
        ) -> torch.Tensor:
        """
        Get the cos angle of the difference between object and TCP
        in the z-axis
        """
        cos_sim_workframe = torch.cos(
            torch.arccos(cos_tcp_Rz_to_goal_workframe) - torch.arccos(cos_obj_Rz_to_goal_workframe)
            )
        return 1 - cos_sim_workframe
    
    def reward(
        self,
        act: torch.Tensor,
        next_obs: torch.Tensor
        ) -> torch.Tensor:
        '''
        Caculate the reward given a batch of observations 
        '''

        if 'reduced' in self.observation_mode: 
            # tcp_pos_to_goal_workframe = next_obs[:, 0:3]
            tcp_orn_to_goal_workframe = next_obs[:, 3:7]
            # tcp_lin_vel_workframe = next_obs[:, 7:10]
            # tcp_ang_vel_workframe = next_obs[:, 10:13]
            cur_obj_pos_to_goal_workframe = next_obs[:, 13:16]
            cur_obj_orn_to_goal_workframe = next_obs[:, 16:20]
            # cur_obj_lin_vel_workframe = next_obs[:, 20:23]
            # cur_obj_ang_vel_workframe = next_obs[:, 23:26]

            obj_goal_pos_dist = self.get_pos_dist(cur_obj_pos_to_goal_workframe)
            obj_goal_orn_dist = self.get_orn_dist(cur_obj_orn_to_goal_workframe)
            tip_obj_orn_dist = self.cos_tcp_dist_to_obj(cur_obj_orn_to_goal_workframe, tcp_orn_to_goal_workframe)

        elif 'tactile_pose' in self.observation_mode: 
            # tcp_pos_to_goal_workframe = next_obs[:, 0:3]
            tcp_orn_to_goal_workframe = next_obs[:, 3:7]
            cur_obj_pos_to_goal_workframe = next_obs[:, 7:10]
            cur_obj_orn_to_goal_workframe = next_obs[:, 10:14]

            obj_goal_pos_dist = self.get_pos_dist(cur_obj_pos_to_goal_workframe)
            obj_goal_orn_dist = self.get_orn_dist(cur_obj_orn_to_goal_workframe)
            tip_obj_orn_dist = self.cos_tcp_dist_to_obj(cur_obj_orn_to_goal_workframe, tcp_orn_to_goal_workframe)

        else:
            # tcp_pos_workframe = next_obs[:, 0:3]
            tcp_rpy_workframe = next_obs[:, 3:6]
            # tcp_lin_vel_workframe = next_obs[:, 6:9]
            # tcp_ang_vel_workframe = next_obs[:, 9:12]
            cur_obj_pos_workframe = next_obs[:, 12:15]
            cur_obj_rpy_workframe = next_obs[:, 15:18]
            # cur_obj_lin_vel_workframe = next_obs[:, 18:21]
            # cur_obj_ang_vel_workframe = next_obs[:, 21:24]
            # pred_goal_pos_workframe = next_obs[:, 24:27]
            # pred_goal_rpy_workframe = next_obs[:, 27:30]

            obj_goal_pos_dist = self.xyz_obj_dist_to_goal(cur_obj_pos_workframe)
            obj_goal_orn_dist = self.orn_obj_dist_to_goal(cur_obj_rpy_workframe)
            tip_obj_orn_dist = self.cos_tcp_dist_to_obj(cur_obj_rpy_workframe, tcp_rpy_workframe)

        reward = -(obj_goal_pos_dist + obj_goal_orn_dist + tip_obj_orn_dist)
        reward = reward[:, None]

        return reward

    def render(self, mode="human"):
        pass


    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            self.reset_batch_goals(batch_size)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)
