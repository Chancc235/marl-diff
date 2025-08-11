import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning
from diffuser.models.backbone import TrajectoryEncoder, PatternEncoder
from diffuser.models.decision_transformer import DecisionTransformer

class CCDiffusion(nn.Module):
    """Offline Diffusion-based RL with VAE for action generation"""
    agent_share_parameters = True
    
    def __init__(
        self,
        model,  # Pre-built diffusion backbone model
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        trajectory_latent_dim: int = 128,
        hidden_dim: int = 256,
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        vae_weight: float = 1.0,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        # Offline RL specific parameters
        conservative_weight: float = 1.0,
        awac_weight: float = 1.0,
        use_conservative_loss: bool = True,
        use_behavior_cloning: bool = True,
        bc_weight: float = 1.0,
        q_weight: float = 1.0,  # Weight for Q-learning loss
        policy_weight: float = 0.1,  # Weight for policy optimization loss
        diversity_weight: float = 0.1,  # Weight for diversity loss
        # Model architecture parameters (for backward compatibility)
        backbone_layers: int = 4,
        backbone_hidden_dim: int = 256,
        pattern_latent_dim: int = 64,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        # New parameter for decentralized execution
        decentralized: bool = True,
        use_ddim_sample: bool = True,
        action_latent_dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.trajectory_latent_dim = trajectory_latent_dim
        self.vae_weight = vae_weight
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.decentralized = decentralized
        
        # Store discrete_action flag for proper action handling
        self.discrete_action = kwargs.get('discrete_action', False)
        self.discount = kwargs.get('discount', 0.99)
        self.n_ddim_steps = kwargs.get('n_ddim_steps', 15)
        self.target_update_freq = kwargs.get('target_update_freq', 10)
        self.target_update_tau = kwargs.get('target_update_tau', 0.5)
        self.n_target_q_functions = kwargs.get('n_target_q_functions', 2)
        self.action_latent_dim = action_latent_dim
        # Offline RL parameters

        self.awac_weight = awac_weight

        self.use_behavior_cloning = use_behavior_cloning
        self.bc_weight = bc_weight
        self.q_weight = q_weight
        self.policy_weight = policy_weight
        self.diversity_weight = diversity_weight
        
        # Trajectory encoder for conditioning
        # For trajectory encoding, always use the full action space dimension
        # even for discrete actions, since we'll handle the conversion inside
        # In training, we always pass full trajectory with all agents, so use centralized input_dim
        # self.trajectory_encoder = TrajectoryEncoder(
        #     observation_dim=observation_dim,
        #     action_dim=action_dim,  # Always use full action space
        #     history_horizon=max(history_horizon, 1),
        #     n_agents=n_agents,
        #     hidden_dim=hidden_dim,
        #     latent_dim=trajectory_latent_dim,
        #     decentralized=False,  # Always use centralized mode for training
        #     # Sequence model parameters
        #     sequence_model=kwargs.get('trajectory_sequence_model', 'transformer'),
        #     num_layers=kwargs.get('trajectory_num_layers', 2),
        #     num_heads=kwargs.get('trajectory_num_heads', 8),
        #     dropout=kwargs.get('trajectory_dropout', 0.1),
        # )
        self.trajectory_encoder = DecisionTransformer(state_dim=observation_dim, 
                                                      act_dim=action_dim, 
                                                      max_ep_len=max(history_horizon, 1),
                                                      n_head=kwargs.get('trajectory_num_heads', 4),
                                                      n_layer=kwargs.get('trajectory_num_layers', 2),
                                                      dropout=kwargs.get('trajectory_dropout', 0.1),
                                                      n_embd=trajectory_latent_dim,
                                                      hidden_size=trajectory_latent_dim,
                                                      )
        self.pattern_encoder = PatternEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=pattern_latent_dim,
            num_layers=kwargs.get('pattern_num_layers', 2),
            dropout=kwargs.get('pattern_dropout', 0.1),
        )
        
        # Store discrete_action flag for proper action handling
        self.discrete_action = kwargs.get('discrete_action', False)
        
        # Use the provided diffusion backbone model
        self.model = model
        
        # Set returns_condition on the backbone model
        if hasattr(self.model, 'returns_condition'):
            self.model.returns_condition = returns_condition
        
        # Calculate proper input dimensions for Q-function
        # Q-function always sees all agents' data for proper value evaluation
        # even in decentralized execution mode
        q_input_dim = observation_dim * n_agents + action_dim * n_agents + trajectory_latent_dim * n_agents
        pattern_q_input_dim = observation_dim * n_agents + pattern_latent_dim * n_agents + trajectory_latent_dim * n_agents
        
        # # Value function for offline RL (Q-function)
        # 将 q_function 也改成列表，便于实现 ensemble 或 double Q-learning
        self.q_function = [
            nn.Sequential(
                nn.Linear(q_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(self.n_target_q_functions)
        ]
        # Target Q-function for stable training
        # 将 target_q_function 写成列表，便于实现 ensemble 或 double Q-learning
        self.target_q_functions = [
            nn.Sequential(
                nn.Linear(q_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(self.n_target_q_functions)
        ]
        # # Initialize target network
        for i, target_q_function in enumerate(self.target_q_functions):
            target_q_function.load_state_dict(self.q_function[i].state_dict())
        
        self.pattern_q_function = nn.Sequential(
            nn.Linear(pattern_q_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        # Noise scheduler
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.beta = kwargs.get('beta', 0.1)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.use_ddim_sample = use_ddim_sample
        if self.use_ddim_sample:
            self.set_ddim_scheduler(n_ddim_steps=self.n_ddim_steps)
            
        # Loss functions
        self.data_encoder = data_encoder
        
        # Ensure all Q functions are on the same device as the main model
        self._move_q_functions_to_device()

    
    def _move_q_functions_to_device(self):
        """Move all Q functions to the same device as the main model"""
        if hasattr(self, 'model') and self.model is not None:
            # Get device from the main model
            device = next(self.model.parameters()).device
            for i in range(self.n_target_q_functions):
                self.q_function[i] = self.q_function[i].to(device)
                self.target_q_functions[i] = self.target_q_functions[i].to(device)
            self.pattern_q_function = self.pattern_q_function.to(device)
            
            # Also move trajectory encoder if it exists
            if hasattr(self, 'trajectory_encoder'):
                self.trajectory_encoder = self.trajectory_encoder.to(device)

    def encode_trajectory(self, trajectory, agent_idx=None, test_obs=None):
        # 按最后一维切分action和obs，并取agent_idx
        action_dim = self.action_dim
        obs_dim = self.observation_dim
        # trajectory: [batch, history_horizon, n_agents, action_dim + obs_dim]
        actions = trajectory[..., :action_dim]
        obs = trajectory[..., action_dim:action_dim+obs_dim]
        if agent_idx is not None:
            actions = actions[:, :, agent_idx, :]
            obs = obs[:, :, agent_idx, :]

        # 若为测试状态，拼接一个新的obs和全0的action
        if test_obs is not None:
            # test_obs: [batch, obs_dim]
            batch_size = obs.shape[0]
            # 新的action全0
            
            new_action = torch.zeros((batch_size, 1, action_dim), dtype=actions.dtype, device=actions.device)
            test_obs = test_obs[:, agent_idx, :]
            new_obs = test_obs.unsqueeze(1) if test_obs.dim() == 2 else test_obs  # [batch, 1, obs_dim]
            actions = torch.cat([actions, new_action], dim=1)
            obs = torch.cat([obs, new_obs], dim=1)
            obs = obs[:, 1:, :]
            actions = actions[:, 1:, :]
        x = self.trajectory_encoder.get_embeddings(obs, actions)
        return x

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        trajectory_condition: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_timestep: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ):
        """Get diffusion model output"""
        
        # Concatenate latents with trajectory condition
        model_input = torch.cat([x, trajectory_condition], dim=-1)
        # Get model output
        if self.returns_condition and returns is not None:
            epsilon_cond = self.model(
                model_input, t, returns=returns, env_timestep=env_timestep,
                attention_masks=attention_masks, use_dropout=False,
            )
            epsilon = epsilon_cond
            # epsilon_uncond = self.model(
            #     model_input, t, returns=returns, env_timestep=env_timestep,
            #     attention_masks=attention_masks, force_dropout=True,
            # )
            # epsilon = epsilon_uncond + self.condition_guidance_w * (
            #     epsilon_cond - epsilon_uncond
            # )
        else:
            epsilon = self.model(
                model_input, t, env_timestep=env_timestep,
                attention_masks=attention_masks, use_dropout=use_dropout,
                force_dropout=force_dropout,
            )
        
        return epsilon
    
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        obs = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_diffusion: bool = False,
        agent_idx: Optional[int] = None,  # New parameter for decentralized mode
        enable_grad: bool = False,
    ):
        """Sample action latents conditioned on trajectory"""
        batch_size = cond["history_trajectory"].shape[0]
        # Use context manager to control gradients
        context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context_manager:
            if "history_trajectory" in cond:
                trajectory_conditions = []
                for agent_idx in range(self.n_agents):
                    trajectory_condition = self.encode_trajectory(cond["history_trajectory"], test_obs=obs, agent_idx=agent_idx)
                    trajectory_conditions.append(trajectory_condition)
                trajectory_condition = torch.stack(trajectory_conditions, dim=1).view(batch_size, self.n_agents, -1)
                batch_size = cond["history_trajectory"].shape[0]
            
                shape = (batch_size, self.n_agents, self.action_dim)

            device = trajectory_condition.device
            if self.use_ddim_sample:
                scheduler = self.ddim_noise_scheduler
            else:
                scheduler = self.noise_scheduler
                
            x = torch.randn(shape, device=device)
            
            # Enable gradients for the initial noise if needed
            if enable_grad:
                x.requires_grad_(True)
            
            if return_diffusion:
                diffusion = [x]
                
            timesteps = scheduler.timesteps
            
            progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
            for t in timesteps:
                ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = self.get_model_output(
                    x, ts, trajectory_condition, returns, env_ts, attention_masks
                )
                
                x = scheduler.step(model_output, t, x).prev_sample
                
                progress.update({"t": t})
                if return_diffusion:
                    diffusion.append(x)
                    
            progress.close()
            # this is logits of action when action is discrete
            actions = x
            
            if return_diffusion:
                diffusion = torch.stack(diffusion, dim=1)
                # Process each agent separately through pattern encoder
                pattern_latents_list = []
                for agent_idx in range(self.n_agents):
                    agent_diffusion = diffusion[:, :, agent_idx, :]  # [batch, seq, action_dim]
                    agent_pattern_latent = self.pattern_encoder(agent_diffusion)  # [batch, latent_dim]
                    pattern_latents_list.append(agent_pattern_latent)
                
                # Stack to get [batch, n_agents, latent_dim]
                pattern_latents = torch.stack(pattern_latents_list, dim=1)
                return actions, pattern_latents, trajectory_condition
            else:
                return actions
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        trajectory_condition: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
    ):
        """Compute diffusion losses"""
        noise = torch.randn_like(x_start)
        
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = self.data_encoder(x_noisy)
        # print("x_noisy",x_noisy.shape)
        # print("t",t.shape)
        # print("trajectory_condition",trajectory_condition.shape)
        # print("returns",returns.shape)
        # print("env_ts",env_ts.shape)
        # print("attention_masks",attention_masks.shape)
        epsilon = self.get_model_output(
            x_noisy, t, trajectory_condition, returns, env_ts, attention_masks
        )
        
        if not self.predict_epsilon:
            epsilon = self.data_encoder(epsilon)
            
        assert noise.shape == epsilon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(epsilon, noise)
        else:
            loss = F.mse_loss(epsilon, x_start)
        
        info = {"diffusion_loss": loss.item()}
            
        return loss, info

    
    def behavioral_cloning_loss(self, obs, actions, trajectory_condition):
        """Behavioral cloning loss to stay close to data distribution"""
        # Generate actions from current policy (with gradients enabled)
        # Since we already have encoded trajectory_condition, we'll do the sampling manually
        batch_size = actions.shape[0]
        device = actions.device
        target_actions = actions

        # Enable gradients for BC loss
        with torch.enable_grad():
            shape = (batch_size, self.n_agents, self.action_dim)

            if self.use_ddim_sample:
                scheduler = self.ddim_noise_scheduler
            else:
                scheduler = self.noise_scheduler
                
            x = torch.randn(shape, device=device)
            timesteps = scheduler.timesteps
            
            for t in timesteps:
                ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = self.get_model_output(
                    x, ts, trajectory_condition, None, None, None
                )
                x = scheduler.step(model_output, t, x).prev_sample
            generated_actions = x
            # For discrete actions, generated_actions are logits, don't apply softmax for cross entropy
        bc_loss = F.mse_loss(generated_actions, target_actions)
        
        return bc_loss
    
    def q_learning_loss(self, obs, actions, rewards, next_obs, next_actions, dones, history_trajectory):
        """Q-learning loss for value function training"""
        
        batch_size = obs.shape[0]
        device = obs.device
        cond = {"history_trajectory": history_trajectory}
        
        # Encode trajectory for each agent and average the conditions
        trajectory_conditions = []
        pattern_latents_list = []
        for agent_idx in range(self.n_agents):
            agent_condition = self.encode_trajectory(history_trajectory, agent_idx=agent_idx)
            trajectory_conditions.append(agent_condition)
        trajectory_conditions = torch.stack(trajectory_conditions, dim=1).view(batch_size, -1).detach()

        obs_last = obs
        actions_last = actions
        next_obs_last = next_obs
        rewards_last = rewards
        dones_last = dones

        # Use all agents data directly
        obs_flat = obs_last.view(batch_size, -1)
        actions_flat = actions_last.view(batch_size, -1)
        next_obs_flat = next_obs_last.view(batch_size, -1)

        # Compute current Q-values
        q_input = torch.cat([obs_flat, actions_flat, trajectory_conditions], dim=-1)
        q_current = [self.q_function[i](q_input) for i in range(self.n_target_q_functions)]
        
        # Compute target Q-values with improved target policy
        with torch.no_grad():
            # Create updated trajectory for next state
            current_step = torch.cat([obs_last, actions_last], dim=-1)  # [batch, n_agents, obs_dim + action_dim]
            current_step = current_step.unsqueeze(1)  # [batch, 1, n_agents, obs_dim + action_dim]
            
            # Update trajectory by appending current step and removing oldest if needed
            updated_trajectory = torch.cat([history_trajectory, current_step], dim=1)  # [batch, horizon+1, n_agents, obs_dim + action_dim]
            if updated_trajectory.shape[1] > self.history_horizon:
                updated_trajectory = updated_trajectory[:, -self.history_horizon:]
            
            # Generate next actions with temperature sampling for better exploration
            next_cond = {"history_trajectory": updated_trajectory}
            actions_next_flat = next_actions.view(batch_size, -1)

            # Concatenate actions and average pool conditions
            next_conditions_list = []
            for agent_idx in range(self.n_agents):
                next_agent_condition = self.encode_trajectory(updated_trajectory, agent_idx=agent_idx)
                next_conditions_list.append(next_agent_condition)
            
            next_trajectory_condition = torch.stack(next_conditions_list, dim=1).view(batch_size, -1)
            next_q_input = torch.cat([next_obs_flat, actions_next_flat, next_trajectory_condition], dim=1)
            next_q = [self.target_q_functions[i](next_q_input).detach() for i in range(self.n_target_q_functions)]
            q_nexts = torch.stack(next_q, dim=0)  # [ensemble_size, batch_size]
            q_mean = q_nexts.mean(dim=0)
            q_std = q_nexts.std(dim=0)
            lcb = q_mean - self.beta * q_std
            # Process rewards and dones
            reward_scalar = rewards_last.view(batch_size, -1).mean(dim=-1, keepdim=True)  # (batch_size, 1)
            done_scalar = dones_last.view(batch_size, -1).any(dim=-1, keepdim=True)       # (batch_size, 1)
            target_q = reward_scalar + self.discount * lcb * (1 - done_scalar.float())
        
        q_loss = [F.mse_loss(q_current[i], target_q) for i in range(self.n_target_q_functions)]
        q_loss = sum(q_loss) / self.n_target_q_functions
        return q_loss

    def pattern_q_learning_loss(self, obs, rewards, dones, history_trajectory):
        """Pattern Q-learning loss for value function training"""
        
        batch_size = obs.shape[0]
        device = obs.device
        cond = {"history_trajectory": history_trajectory}

        # Encode trajectory for each agent and average the conditions
        obs_last = obs
        rewards_last = rewards
        dones_last = dones

        obs_flat = obs_last.view(batch_size, -1)
        actions, pattern_latents, trajectory_conditions = self.conditional_sample(cond, return_diffusion=True, verbose=False)
        actions_flat = actions.view(batch_size, -1)
        pattern_latents_flat = pattern_latents.view(batch_size, -1).detach()

        trajectory_conditions = trajectory_conditions.view(batch_size, -1).detach()
        q_input = torch.cat([obs_flat, actions_flat, trajectory_conditions], dim=-1)
        pattern_q_input = torch.cat([obs_flat, pattern_latents_flat, trajectory_conditions], dim=-1)

        q_current = self.pattern_q_function(pattern_q_input)
        q_label = [self.q_function[i](q_input).detach() for i in range(self.n_target_q_functions)]
        q_label = torch.stack(q_label, dim=0)
        q_label = q_label.mean(dim=0) - self.beta * q_label.std(dim=0)
        q_loss = F.mse_loss(q_current, q_label)
        return q_loss
        

    def policy_optimization_loss(self, obs, actions, history_trajectory, agent_idx=None):
        """Policy optimization loss: maximize Q(s, π(s)) + diversity loss"""
        
        batch_size = obs.shape[0]
        device = obs.device
        
        # In decentralized mode, compute loss for each agent separately
        if self.decentralized:
            # Collect all agent actions first
            all_agent_actions_latents = []
            agent_actions_list = []
            cond = {"history_trajectory": history_trajectory}

            # Enable gradients for policy optimization
            agent_actions, pattern_latents, trajectory_conditions = self.conditional_sample(
                cond, return_diffusion=True, enable_grad=True
            )

            policy_actions = agent_actions
            policy_actions_latents = pattern_latents

            # Flatten for Q-function input  
            obs_flat = obs.view(batch_size, -1)
            pattern_latents_flat = policy_actions_latents.view(batch_size, -1)
            trajectory_condition_flat = trajectory_conditions.view(batch_size, -1).detach()
            # Compute current Q-values
            pattern_q_input = torch.cat([obs_flat, pattern_latents_flat, trajectory_condition_flat], dim=1)
            pattern_q_value = self.pattern_q_function(pattern_q_input).detach()
            
            # Use clipped Q-values for more stable training
            pattern_q_value = torch.clamp(pattern_q_value, -10, 10)
            
            policy_loss = -pattern_q_value.mean() + 1e-8

        return policy_loss

    def loss(
        self,
        x: torch.Tensor,  # Full trajectory
        actions: torch.Tensor,  # Actions only
        history_trajectory: torch.Tensor,  # Historical trajectory
        cond: Dict[str, torch.Tensor],
        observations: Optional[torch.Tensor] = None,  # Observations only
        rewards: Optional[torch.Tensor] = None,
        next_observations: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        agent_idx: Optional[int] = None,  # New parameter for decentralized training
        loss_masks: Optional[torch.Tensor] = None,  # 保持兼容性，但不使用
        **kwargs,
    ):
        """Compute total offline RL loss"""
        
        batch_size = len(x)
        actions = actions[:, -1, :, :].squeeze(1)  # [batch, n_agents, action_dim]
        observations = observations[:, -1, :, :].squeeze(1)  # [batch, n_agents, obs_dim]
        # Encode historical trajectory fosr conditioning
        trajectory_conditions = []
        for agent_idx in range(self.n_agents):
            trajectory_condition = self.encode_trajectory(history_trajectory, agent_idx=agent_idx)
            trajectory_conditions.append(trajectory_condition)
        trajectory_condition = torch.stack(trajectory_conditions, dim=1).view(batch_size, self.n_agents, -1)

        # Compute diffusion loss
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=x.device,
        ).long()

        # diffusion_loss, info = self.p_losses(
        #     actions, trajectory_condition, t,
        #     returns, env_ts, attention_masks,
        # )

        # # Initialize total loss
        # total_loss = diffusion_loss
        total_loss = torch.tensor(0.0, device=x.device)
        info = {}
        if self.use_behavior_cloning:
            bc_loss = self.behavioral_cloning_loss(observations, actions, trajectory_condition)
            total_loss += self.bc_weight * bc_loss
            info["bc_loss"] = bc_loss.item() if torch.is_tensor(bc_loss) else bc_loss

        return total_loss, info
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update of target Q-network"""
        for i in range(self.n_target_q_functions):
            for target_param, param in zip(self.target_q_functions[i].parameters(), self.q_function[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_action(self, obs, history_trajectory=None, deterministic=False, agent_idx=None, returns=None):
        """Get action from current policy
        
        Args:
            obs: [batch_size, n_agents, obs_dim] (centralized) or [batch_size, obs_dim] (decentralized)
            history_trajectory: Historical trajectory for conditioning  
            deterministic: Whether to use deterministic sampling
            agent_idx: If provided, only return action for this agent (decentralized execution)
            returns: [batch_size, 1] return values for conditioning (optional)
        """
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            cond = {"history_trajectory": history_trajectory}
            actions = self.conditional_sample(cond, obs, returns=returns, verbose=False)
            return actions

    
    def forward(self, cond, *args, **kwargs):
        """Generate actions using the complete pipeline"""
        return self.conditional_sample(cond, *args, **kwargs)
    
    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        """Set DDIM scheduler for faster sampling"""
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True


    def compute_q_loss(self, x, actions, history_trajectory, cond, observations=None, 
                      rewards=None, next_observations=None, next_actions=None, dones=None, returns=None, 
                      env_ts=None, attention_masks=None, **kwargs):
        """Compute Q-learning loss for all agents together (centralized)"""
        
        batch_size = len(x)
        obs_last = observations[:, -1]  # [batch, n_agents, obs_dim]

        # Q-learning loss if we have next states and rewards
        q_loss = torch.tensor(0.0, device=x.device)
        info = {}
        
        if next_observations is not None and rewards is not None and dones is not None:
            next_obs_last = next_observations[:, -1]
            rewards_last = rewards[:, -1]  # [batch, n_agents, 1]
            dones_last = dones[:, -1]  # [batch, n_agents, 1]
            actions_last = actions[:, -1]  # [batch, n_agents, action_dim]
            next_actions = next_actions[:, -1]  # [batch, n_agents, action_dim]
            # Compute Q-learning loss for all agents (centralized)
            q_loss = self.q_learning_loss(obs_last, actions_last, rewards_last, next_obs_last, next_actions, dones_last, history_trajectory)
            
            info["q_loss"] = q_loss.item() if torch.is_tensor(q_loss) else q_loss
        else:
            info["q_loss"] = 0.0
        
        return q_loss, info

    def compute_pattern_q_loss(self, x, rewards, dones, history_trajectory, cond, observations=None, **kwargs):
        """Compute pattern Q-learning loss for all agents together (centralized)"""
        rewards = rewards[:, -1]  # [batch, n_agents, 1]
        dones = dones[:, -1]  # [batch, n_agents, 1]
        observations = observations[:, -1]  # [batch, n_agents, obs_dim]
        # Compute pattern Q-learning loss for all agents (centralized)
        pattern_q_loss = self.pattern_q_learning_loss(observations, rewards, dones, history_trajectory)
        info = {}
        info["pattern_q_loss"] = pattern_q_loss.item() if torch.is_tensor(pattern_q_loss) else pattern_q_loss
        return pattern_q_loss, info

    def compute_policy_loss(self, x, actions, history_trajectory, cond, observations=None, 
                           rewards=None, next_observations=None, dones=None, returns=None, 
                           env_ts=None, attention_masks=None, **kwargs):
        """Compute policy optimization loss for all agents together (centralized)"""
        
        batch_size = len(x)
    
        # Extract last timestep data (not t0)
        if len(observations.shape) == 4:  # [batch, horizon, n_agents, obs_dim]
            obs_last = observations[:, -1]  # [batch, n_agents, obs_dim]
        else:
            obs_last = observations
        
        # Policy optimization loss
        info = {}
        
        actions_last = actions[:, -1]  # [batch, n_agents, action_dim]

        # Compute policy optimization loss for all agents (centralized)
        policy_loss = self.policy_optimization_loss(obs_last, actions_last, history_trajectory)
        info["policy_loss"] = policy_loss.item() if torch.is_tensor(policy_loss) else policy_loss
        
        return self.policy_weight * policy_loss, info

    def compute_q_and_policy_losses(self, x, actions, history_trajectory, cond, observations=None, 
                                   rewards=None, next_observations=None, dones=None, returns=None, 
                                   env_ts=None, attention_masks=None, **kwargs):
        """Compute Q-learning and policy optimization losses for all agents together (centralized)"""
        
        # Compute Q loss
        pattern_q_loss_weighted, pattern_q_info = self.compute_pattern_q_loss(x, rewards, dones, history_trajectory, cond, observations, **kwargs)
        # Compute policy loss
        policy_loss_weighted, policy_info = self.compute_policy_loss(x, actions, history_trajectory, cond, observations,
                                                                    rewards, next_observations, dones, returns,
                                                                    env_ts, attention_masks, **kwargs)
        
        # Combine info
        info = {**pattern_q_info, **policy_info}
        info = {}
        pattern_q_loss_weighted = torch.tensor(0.0, device=x.device)
        policy_loss_weighted = torch.tensor(0.0, device=x.device)
        return pattern_q_loss_weighted, policy_loss_weighted, info

    def to(self, device):
        """Move model to device and ensure all Q functions are also moved"""
        # Call parent class to method first
        super().to(device)
        
        # Move all Q functions to the same device
        for i in range(self.n_target_q_functions):
            self.q_function[i] = self.q_function[i].to(device)
            self.target_q_functions[i] = self.target_q_functions[i].to(device)
        self.pattern_q_function = self.pattern_q_function.to(device)
        
        # Also move trajectory encoder if it exists
        if hasattr(self, 'trajectory_encoder'):
            self.trajectory_encoder = self.trajectory_encoder.to(device)
            
        return self
