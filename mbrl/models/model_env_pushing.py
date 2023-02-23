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

from mbrl.util.math import euler_to_quaternion, quaternion_rotation_matrix, get_orn_diff
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
        self.terminated_early_penalty = env.terminated_early_penalty
        self.reached_goal_reward = env.reached_goal_reward 
        self.traj_n_points = env.traj_n_points
        self.TCP_lims = env.robot.arm.TCP_lims
        self.max_tcp_to_obj_orn = env.max_tcp_to_obj_orn #- (0/180 * torch.pi)
        self.max_tip_to_obj_pos = 0.02
        self.task = env.task
        self.planar_states = env.planar_states
        self.mpc_goal_orn_update = env.mpc_goal_orn_update
  
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
            # Update goal orn in mpc only if current obs outside of sensitive zone
            if self.mpc_goal_orn_update: 
                self.update_step_data(next_observs)
                
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs, rewards)

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
        batch_size: int,
        sample_goals: bool = False,
        goal_batch: Optional[np.ndarray] = np.array([]),
        ):
        '''
        Set the goal batches and target_ids to match with current goals of the gym
        environment.
        '''

        if sample_goals:
            self.goal_pos_workframe_batch = np.empty((batch_size, 3))
            self.goal_pos_workframe_batch[:, 0:2] = self.sample_batch_goals(batch_size)
            self.goal_pos_workframe_batch = model_util.to_tensor(self.goal_pos_workframe_batch).to(torch.float32).to(self.device)
        
        else:
            if goal_batch.size != 0:
                self.goal_pos_workframe_batch = model_util.to_tensor(goal_batch).to(torch.float32).to(self.device)
            else:   
                # Initialise variables from gym environments
                self.targ_traj_list_id = torch.tensor(self.env.targ_traj_list_id).to(self.device)
                if self.targ_traj_list_id < 0:
                    print("Error targ_traj_list_id is below 0, the gym environment has not been reset.")
                
                self.traj_pos_workframe = model_util.to_tensor(copy.deepcopy(self.env.traj_pos_workframe)).to(torch.float32).to(self.device)
                self.traj_orn_workframe = model_util.to_tensor(copy.deepcopy(self.env.traj_orn_workframe)).to(torch.float32).to(self.device)

                self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
                if self.task == "goal_pos":
                    self.goal_orn_workframe = model_util.to_tensor(copy.deepcopy(self.env.goal_orn_workframe)).to(torch.float32).to(self.device)
                else:
                    self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id]

                # Create the batches needed to evaluate actions
                self.goal_pos_workframe_batch = torch.tile(self.goal_pos_workframe, 
                                                (batch_size,) 
                                                + tuple([1] * self.goal_pos_workframe.ndim))
                self.goal_orn_workframe_batch = torch.tile(self.goal_orn_workframe, 
                                                (batch_size,) 
                                                + tuple([1] * self.goal_orn_workframe.ndim))

        # Get reward weights
        self.W_obj_goal_pos = torch.tensor(self.env.W_obj_goal_pos).to(self.device)
        self.W_obj_goal_orn = torch.tensor(self.env.W_obj_goal_orn).to(self.device)
        self.W_tip_obj_orn = torch.tensor(self.env.W_tip_obj_orn).to(self.device)
        self.W_act = torch.tensor(self.env.W_act).to(self.device)

        self.W_obj_goal_pos_batch = torch.tile(self.W_obj_goal_pos, (batch_size,))
        self.W_obj_goal_orn_batch = torch.tile(self.W_obj_goal_orn, (batch_size,))
        self.W_tip_obj_orn_batch = torch.tile(self.W_tip_obj_orn, (batch_size,))
        self.W_act_batch = torch.tile(self.W_act, (batch_size,))
        
    def outside_tcp_lims(self, cur_obj_pos_workframe, tcp_to_obj_orn, tip_to_obj_pos):
        '''
        Function to check if either object or tcp postion is outside of the 
        TCP limit
        '''
        # xyz_tcp_dist_to_obj = torch.linalg.norm(tcp_pos_workframe - cur_obj_pos_workframe)
        return ((cur_obj_pos_workframe[:, 0] < self.TCP_lims[0, 0]) | 
                (cur_obj_pos_workframe[:, 0] > self.TCP_lims[0, 1]) | 
                (cur_obj_pos_workframe[:, 1] < self.TCP_lims[1, 0]) | 
                (cur_obj_pos_workframe[:, 1] > self.TCP_lims[1, 1]) |
                (tcp_to_obj_orn > self.max_tcp_to_obj_orn) | 
                (tip_to_obj_pos > self.max_tip_to_obj_pos))
    
    def sample_batch_goals(
        self, 
        batch_size: int
    ):
        return np.array([self.env.random_single_goal() for _ in range(batch_size)])

    def get_agent_obs(
        self,
        obs: np.ndarray,
        goal: Optional[np.ndarray] = np.array([]),
    ) -> np.ndarray:

        if 'goal_aware' in self.observation_mode:
            return obs

        agent_obs = torch.zeros(obs.shape, dtype=torch.float32).to(self.device)
        agent_obs[:, 0:4] = model_util.to_tensor(obs[:, 0:4]).to(torch.float32).to(self.device)

        cur_obj_pos_workframe = torch.zeros((len(obs), 3), dtype=torch.float32).to(self.device)
        cur_obj_orn_workframe = torch.zeros((len(obs), 4), dtype=torch.float32).to(self.device)
        cur_obj_pos_workframe[:, 0:2] = model_util.to_tensor(obs[:, 4:6]).to(torch.float32).to(self.device)
        cur_obj_orn_workframe[:, 2:4] = model_util.to_tensor(obs[:, 6:8]).to(torch.float32).to(self.device)

        if goal.size != 0:
            goal_pos_workframe = model_util.to_tensor(goal).to(torch.float32).to(self.device)
            goal_rpy_workframe = torch.zeros((len(obs), 3), dtype=torch.float32).to(self.device)
            goal_rpy_workframe[:, 2] = torch.atan2(-cur_obj_pos_workframe[:, 1] + goal_pos_workframe[:, 1], 
                                                        -cur_obj_pos_workframe[:, 0] + goal_pos_workframe[:, 0])
            goal_orn_workframe = euler_to_quaternion(goal_rpy_workframe)

            cur_obj_to_goal_pos_workframe = cur_obj_pos_workframe - goal_pos_workframe
            cur_obj_to_goal_orn_workframe = get_orn_diff(cur_obj_orn_workframe, goal_orn_workframe)
        else:
            cur_obj_to_goal_pos_workframe = cur_obj_pos_workframe - self.goal_pos_workframe_batch
            cur_obj_to_goal_orn_workframe = get_orn_diff(cur_obj_orn_workframe, self.goal_orn_workframe_batch)

        agent_obs[:, 4:6] = cur_obj_to_goal_pos_workframe[:, 0:2]
        agent_obs[:, 6:8] = cur_obj_to_goal_orn_workframe[:, 2:4]

        return agent_obs.cpu().numpy()

    def update_step_data(
        self,
        next_obs: torch.Tensor
    ):
        if isinstance(next_obs, torch.Tensor):
            obs = next_obs
        else:
            obs = model_util.to_tensor(next_obs).to(torch.float32).to(self.device)

        if self.observation_mode == 'tactile_pose_data' or self.observation_mode == 'tactile_pose_relative_data':
            if self.planar_states == True:
                goal_rpy_workframe_batch = torch.zeros((len(obs), 3), dtype=torch.float32).to(self.device)
                goal_rpy_workframe_batch[:, 2] = torch.atan2(-obs[:, 5] + self.goal_pos_workframe_batch[:, 1], 
                                                            -obs[:, 4] + self.goal_pos_workframe_batch[:, 0])

                self.goal_orn_workframe_batch = euler_to_quaternion(goal_rpy_workframe_batch)
        else:
            NotImplemented
        
    def termination(
        self, 
        act: torch.Tensor, 
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
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

        elif self.observation_mode == 'goal_aware_tactile_pose_data': 
            if self.planar_states == True: 
                # tcp_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                # tcp_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                # cur_obj_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                # tcp_pos_to_goal_workframe[:, 0:2] = next_obs[:, 0:2]
                # tcp_orn_to_goal_workframe[:, 2:4] = next_obs[:, 0:2]
                cur_obj_pos_to_goal_workframe[:, 0:2]= next_obs[:, 2:4]
                # cur_obj_orn_to_goal_workframe[:, 2:4] = next_obs[:, 4:6]
            else:
                # tcp_pos_to_goal_workframe = next_obs[:, 0:3]
                # tcp_orn_to_goal_workframe = next_obs[:, 0:4]
                cur_obj_pos_to_goal_workframe = next_obs[:, 4:7]
                # cur_obj_orn_to_goal_workframe = next_obs[:, 7:11]

            # tcp_pos_workframe = tcp_pos_to_goal_workframe[:, 0:2] + self.goal_pos_workframe_batch[:, 0:2]
            cur_obj_pos_workframe = cur_obj_pos_to_goal_workframe[:, 0:2] + self.goal_pos_workframe_batch[:, 0:2]

             # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_to_goal_workframe, axis=1)

        elif self.observation_mode == 'tactile_pose_data':

            if self.planar_states == True:
                tcp_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                tcp_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_workframe[:, 0:2]= next_obs[:, 4:6]
                cur_obj_orn_workframe[:, 2:4] = next_obs[:, 6:8]
            else:   

                # tcp_pos_workframe = next_obs[:, 0:3]
                tcp_orn_workframe = next_obs[:, 0:4]
                cur_obj_pos_workframe = next_obs[:, 4:7]
                cur_obj_orn_workframe = next_obs[:, 7:11]

            # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_workframe - self.goal_pos_workframe_batch, axis=1)

            # calculate tcp to object orn
            abs_tcp_to_obj_orn = self.get_orn_dist(cur_obj_orn_workframe, tcp_orn_workframe)

            # Calculate tip to obj distance 
            tip_to_obj_pos = torch.linalg.norm(tcp_pos_workframe - cur_obj_pos_workframe, axis=1)

        elif self.observation_mode == 'tactile_pose_relative_data':

            if self.planar_states == True:
                tcp_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                tcp_obj_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_obj_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_workframe[:, 0:2] = next_obs[:, 4:6]
                cur_obj_orn_workframe[:, 2:4] = next_obs[:, 6:8]
            else:   
                NotImplemented

            # Calculate distance between goal and current positon
            obj_goal_pos_dist = self.xyz_obj_dist_to_goal(cur_obj_pos_workframe)

            # calculate tcp to object orn
            abs_tcp_to_obj_orn = self.get_orn_norm(tcp_obj_orn_workframe)

            # Calculate tip to obj distance 
            tip_to_obj_pos = torch.linalg.norm(tcp_obj_pos_workframe, axis=1)

        elif self.observation_mode == 'goal_aware_tactile_pose_relative_data':

            if self.planar_states == True:
                tcp_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                tcp_obj_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_obj_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_to_goal_workframe[:, 0:2] = next_obs[:, 4:6]
                cur_obj_orn_to_goal_workframe[:, 2:4] = next_obs[:, 6:8]
            else:   
                NotImplemented

            # Calculate distance between goal and current positon
            obj_goal_pos_dist = torch.linalg.norm(cur_obj_pos_to_goal_workframe, axis=1)

            # calculate tcp to object orn
            abs_tcp_to_obj_orn = self.get_orn_norm(tcp_obj_orn_workframe)

            # Calculate tip to obj distance 
            tip_to_obj_pos = torch.linalg.norm(tcp_obj_pos_workframe, axis=1)

            # Calculate distance between goal and current obj positon
            cur_obj_pos_workframe = cur_obj_pos_to_goal_workframe[:, 0:2] + self.goal_pos_workframe_batch[:, 0:2]
            
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

        # Create terminated vector, terminated is true if last subgoal is reached
        terminated = torch.zeros((batch_size, 1), dtype=bool).to(self.device)
        terminated[obj_goal_pos_dist < self.termination_pos_dist] = True
        rewards[obj_goal_pos_dist < self.termination_pos_dist] += self.reached_goal_reward

        # Early termination if outside of the tcp limits
        if self.terminate_early:
            outside_tcp_lims_idx = self.outside_tcp_lims(cur_obj_pos_workframe, abs_tcp_to_obj_orn, tip_to_obj_pos)
            terminated[outside_tcp_lims_idx] = True
            rewards[outside_tcp_lims_idx] += self.terminated_early_penalty        # Add a penalty for exiting the TCP limits
    
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

    def get_pos_diff(self, pos_1, pos_2):
        """
        Compute the L2 distance
        """
        dist = torch.linalg.norm(pos_1 - pos_2, axis=1)
        return dist

    def orn_obj_dist_to_goal(
        self, 
        obj_orn_data: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate distance (angle) between the current obj orientation and goal orientation. 
        """

        # obj to goal orientation
        if self.observation_mode == 'tactile_pose_data' or self.observation_mode == 'tactile_pose_relative_data':
            cur_obj_orn_workframe = obj_orn_data
        else:
            cur_obj_orn_workframe = euler_to_quaternion(obj_orn_data)
        inner_product = torch.sum(self.goal_orn_workframe_batch*cur_obj_orn_workframe, 1)
        return torch.arccos(torch.clip(2 * (inner_product ** 2) - 1, -1, 1))

    def get_orn_norm(
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

    def get_orn_dist(
        self, 
        obj_orn_data_1: torch.Tensor, 
        obj_orn_data_2: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Calculate the distance (angle) between two quarternion
        """
        inner_product = torch.sum(obj_orn_data_1 * obj_orn_data_2, 1)
        return torch.arccos(torch.clip(2 * (inner_product ** 2) - 1, -1, 1))

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
        if 'reduced' in self.observation_mode: 
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
            obj_goal_orn_dist = self.get_orn_norm(cur_obj_orn_to_goal_workframe)
            tip_obj_orn_dist = self.cos_tcp_dist_to_obj(cur_obj_orn_to_goal_workframe, tcp_orn_to_goal_workframe)

        elif self.observation_mode == 'goal_aware_tactile_pose_data': 

            if self.planar_states == True: 
                # tcp_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                # tcp_pos_to_goal_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_orn_to_goal_workframe[:, 2:4] = next_obs[:, 0:2]
                cur_obj_pos_to_goal_workframe[:, 0:2]= next_obs[:, 2:4]
                cur_obj_orn_to_goal_workframe[:, 2:4] = next_obs[:, 4:6]
            else:
                # tcp_pos_to_goal_workframe = next_obs[:, 0:3]
                tcp_orn_to_goal_workframe = next_obs[:, 0:4]
                cur_obj_pos_to_goal_workframe = next_obs[:, 4:7]
                cur_obj_orn_to_goal_workframe = next_obs[:, 7:11]

            obj_goal_pos_dist = self.get_pos_dist(cur_obj_pos_to_goal_workframe)
            obj_goal_orn_dist = self.get_orn_norm(cur_obj_orn_to_goal_workframe)
            tip_obj_orn_dist = self.get_orn_dist(cur_obj_orn_to_goal_workframe, tcp_orn_to_goal_workframe)

        elif self.observation_mode == 'tactile_pose_data':

            if self.planar_states == True:
                # tcp_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                # tcp_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_workframe[:, 0:2]= next_obs[:, 4:6]
                cur_obj_orn_workframe[:, 2:4] = next_obs[:, 6:8]
            else:   
                # tcp_pos_workframe = next_obs[:, 0:3]
                tcp_orn_workframe = next_obs[:, 0:4]
                cur_obj_pos_workframe = next_obs[:, 4:7]
                cur_obj_orn_workframe = next_obs[:, 7:11]

            obj_goal_pos_dist = self.xyz_obj_dist_to_goal(cur_obj_pos_workframe)
            obj_goal_orn_dist = self.orn_obj_dist_to_goal(cur_obj_orn_workframe)
            tip_obj_orn_dist = self.get_orn_dist(cur_obj_orn_workframe, tcp_orn_workframe)

        elif self.observation_mode == 'tactile_pose_relative_data':

            if self.planar_states == True:

                tcp_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                tcp_obj_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_obj_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_workframe[:, 0:2] = next_obs[:, 4:6]
                cur_obj_orn_workframe[:, 2:4] = next_obs[:, 6:8]

            else:
                NotImplemented

            obj_goal_pos_dist = self.xyz_obj_dist_to_goal(cur_obj_pos_workframe)
            obj_goal_orn_dist = self.orn_obj_dist_to_goal(cur_obj_orn_workframe)
            tip_obj_orn_dist = self.get_orn_norm(tcp_obj_orn_workframe)

        elif self.observation_mode == 'goal_aware_tactile_pose_relative_data':

            if self.planar_states == True:
                tcp_obj_pos_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                tcp_obj_orn_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)
                cur_obj_pos_to_goal_workframe = torch.zeros((len(next_obs), 3), dtype=torch.float32).to(self.device)
                cur_obj_orn_to_goal_workframe = torch.zeros((len(next_obs), 4), dtype=torch.float32).to(self.device)

                tcp_obj_pos_workframe[:, 0:2] = next_obs[:, 0:2]
                tcp_obj_orn_workframe[:, 2:4] = next_obs[:, 2:4]
                cur_obj_pos_to_goal_workframe[:, 0:2] = next_obs[:, 4:6]
                cur_obj_orn_to_goal_workframe[:, 2:4] = next_obs[:, 6:8]
            else:   
                NotImplemented

            obj_goal_pos_dist = self.get_pos_dist(cur_obj_pos_to_goal_workframe)
            obj_goal_orn_dist = self.get_orn_norm(cur_obj_orn_to_goal_workframe)
            tip_obj_orn_dist = self.get_orn_norm(tcp_obj_orn_workframe)
        
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

        reward_act = torch.sum(torch.square(act), axis=1)

        if self.env.additional_reward_settings == 'john_guide_off_normal':
            approach_zone_idx = obj_goal_pos_dist < 0.1
            self.W_obj_goal_pos_batch[approach_zone_idx] = self.env.importance_obj_goal_pos
            self.W_obj_goal_orn_batch[approach_zone_idx] = 0.0 
            self.W_tip_obj_orn_batch[approach_zone_idx] = self.env.importance_obj_goal_pos / 5.0

        reward = -(
            (self.W_obj_goal_pos_batch * obj_goal_pos_dist)
            + (self.W_obj_goal_orn_batch * obj_goal_orn_dist)
            + (self.W_tip_obj_orn_batch * tip_obj_orn_dist)
            + (self.W_act_batch * reward_act))
            
        reward = reward[:, None]

        return reward

    def render(self, mode="human"):
        pass


    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        initial_action: np.ndarray = None,
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
            if type(initial_action) != np.ndarray:
                initial_action = np.zeros(action_dim)
            tiling_shape_state = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )   
            tiling_shape_action = (num_particles * population_size,) + tuple(
                [1] * initial_action.ndim
            )   
            initial_obs_batch = np.tile(initial_state, tiling_shape_state).astype(np.float32)
            previous_act_batch =  np.tile(initial_action, tiling_shape_action).astype(np.float32)
            previous_act_batch = model_util.to_tensor(previous_act_batch).to(torch.float32).to(self.device)
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
                action_batch = torch.cat([previous_act_batch, action_batch], dim=action_batch.ndim - 1)
                action_batch = action_batch[..., action_sequences.shape[-1]:]
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                previous_act_batch = action_batch
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)

    def rollout_model_env(
        self,
        initial_state: np.ndarray,
        action_sequences: torch.tensor,
        num_particles: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rolls out an environment model.

        Executes a plan on a dynamics model.

        Args:
            model_env (:class:`mbrl.models.ModelEnv`): the dynamics model environment to simulate.
            initial_obs (np.ndarray): initial observation to start the episodes.
            plan (np.ndarray, optional): sequence of actions to execute.
            agent (:class:`mbrl.planning.Agent`): an agent to generate a plan before
                execution starts (as in `agent.plan(initial_obs)`). If given, takes precedence
                over ``plan``.
            num_samples (int): how many samples to take from the model (i.e., independent rollouts).
                Defaults to 1.

        Returns:
            (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.

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
            model_state = self.reset(initial_obs_batch, return_as_np=True)
            batch_size = initial_obs_batch.shape[0]
            self.reset_batch_goals(batch_size)

            obs_history = []
            reward_history = []
            obs_history.append(initial_obs_batch)           
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                next_obs, reward, dones, model_state = self.step(
                    action_batch, model_state, sample=False
                )
                obs_history.append(next_obs)
                reward_history.append(reward)
                
        return np.stack(obs_history), np.stack(reward_history)

