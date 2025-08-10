import argparse
import os

import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # configs that does not exist in old yaml files
    Config.discrete_action = getattr(Config, "discrete_action", False)
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None)
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None)
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    Config.residual_attn = getattr(Config, "residual_attn", True)
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True)
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False)
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
    Config.joint_inv = getattr(Config, "joint_inv", False)
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True)
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)
    
    # VAE+Diffusion RL specific configs
    Config.trajectory_latent_dim = getattr(Config, "trajectory_latent_dim", 128)
    Config.vae_weight = getattr(Config, "vae_weight", 1.0)
    Config.rl_learning_rate = getattr(Config, "rl_learning_rate", 3e-4)
    Config.reward_scale = getattr(Config, "reward_scale", 1.0)
    Config.policy_gradient_weight = getattr(Config, "policy_gradient_weight", 1.0)
    Config.value_function_weight = getattr(Config, "value_function_weight", 0.5)
    Config.entropy_weight = getattr(Config, "entropy_weight", 0.01)
    Config.use_value_function = getattr(Config, "use_value_function", True)
    Config.replay_buffer_size = getattr(Config, "replay_buffer_size", 100000)
    Config.target_update_freq = getattr(Config, "target_update_freq", 1000)
    Config.online_training = getattr(Config, "online_training", False)
    Config.rl_warmup_steps = getattr(Config, "rl_warmup_steps", 10000)
    Config.rl_collect_freq = getattr(Config, "rl_collect_freq", 1000)
    Config.rl_collect_steps = getattr(Config, "rl_collect_steps", 100)
    Config.rl_update_freq = getattr(Config, "rl_update_freq", 100)
    Config.offline_rl_weight = getattr(Config, "offline_rl_weight", 1.0)
    Config.online_rl_weight = getattr(Config, "online_rl_weight", 1.0)
    
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution:
            Config.agent_condition_type = "single"
        else:
            Config.agent_condition_type = "all"

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    
    # Use VAE-specific dataset
    dataset_config = utils.Config(
        "datasets.VAESequenceDataset",  # Use VAE dataset
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        include_returns=Config.returns_condition,
        include_env_ts=Config.env_ts_condition,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        **dataset_extra_kwargs,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath="render_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
    )
    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl",
    )

    dataset = dataset_config()
    renderer = render_config()
    data_encoder = data_encoder_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    action_latent_dim = Config.action_latent_dim
    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    
    # Create base model config (following train.py pattern)
    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        n_agents=Config.n_agents,
        # For DiffusionBackbone, we need different parameters than TemporalUnet
        action_dim=action_dim,
        trajectory_latent_dim=Config.trajectory_latent_dim,
        horizon=Config.horizon,
        hidden_dim=getattr(Config, 'backbone_hidden_dim', Config.hidden_dim),
        n_layers=getattr(Config, 'backbone_layers', 4),
        time_embed_dim=getattr(Config, 'time_embed_dim', 128),
        decentralized=getattr(Config, 'decentralized_execution', False),
        returns_condition=getattr(Config, 'returns_condition', False),
    )

    # Create diffusion model config (following train.py pattern)
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        trajectory_latent_dim=Config.trajectory_latent_dim,
        hidden_dim=Config.hidden_dim,
        n_timesteps=Config.n_diffusion_steps,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        vae_weight=Config.vae_weight,
        discount=Config.discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        conservative_weight=getattr(Config, 'conservative_weight', 1.0),
        awac_weight=getattr(Config, 'awac_weight', 1.0),
        use_conservative_loss=getattr(Config, 'use_conservative_loss', True),
        use_behavior_cloning=getattr(Config, 'use_behavior_cloning', True),
        bc_weight=getattr(Config, 'bc_weight', 1.0),
        q_weight=getattr(Config, 'q_weight', 1.0),  # Weight for Q-learning loss
        policy_weight=getattr(Config, 'policy_weight', 0.1),  # Weight for policy optimization loss
        diversity_weight=getattr(Config, 'diversity_weight', 0.1),  # Weight for diversity loss
        backbone_layers=getattr(Config, 'backbone_layers', 4),
        backbone_hidden_dim=getattr(Config, 'backbone_hidden_dim', 256),
        pattern_latent_dim=getattr(Config, 'pattern_latent_dim', 64),
        data_encoder=data_encoder,
        decentralized=getattr(Config, 'decentralized_execution', False),
        discrete_action=getattr(Config, 'discrete_action', False),  # Add discrete_action parameter
        n_ddim_steps=getattr(Config, 'n_ddim_steps', 15),
        action_latent_dim=getattr(Config, 'action_latent_dim', 16),
    )

    # Choose trainer based on diffusion type (offline only)
    if Config.diffusion == "models.CCDiffusion":
        trainer_class = utils.CCRLTrainer
        logger.print("Using CCRLTrainer for offline RL training", color="blue")
    else:
        trainer_class = utils.Trainer
        logger.print("Using standard Trainer for offline-only training", color="blue")

    # Configure trainer parameters based on trainer type
    trainer_params = {
        "savepath": "trainer_config.pkl",
        "train_batch_size": Config.batch_size,
        "train_lr": Config.learning_rate,
        "gradient_accumulate_every": Config.gradient_accumulate_every,
        "ema_decay": Config.ema_decay,
        "sample_freq": Config.sample_freq,
        "save_freq": Config.save_freq,
        "log_freq": Config.log_freq,
        "label_freq": int(Config.n_train_steps // Config.n_saves),
        "eval_freq": Config.eval_freq,
        "save_parallel": Config.save_parallel,
        "bucket": logger.root,
        "n_reference": Config.n_reference,
        "train_device": Config.device,
        "save_checkpoints": Config.save_checkpoints,
        "update_ema_every": Config.update_ema_every,
        "q_lr": Config.q_lr,
        "q_pattern_lr": Config.q_pattern_lr,
        "q_pretrain_steps": Config.q_pretrain_steps,
    }
    
    # Add trainer-specific parameters
    if trainer_class == utils.CCRLTrainer:
        # Offline RL-specific parameters
        trainer_params.update({
            "target_update_freq": getattr(Config, 'target_update_freq', 5),
        })

    trainer_config = utils.Config(trainer_class, **trainer_params)

    evaluator_config = utils.Config(
        Config.evaluator,
        savepath="evaluator_config.pkl",
        verbose=True,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    # Create model first
    model = model_config()
    
    # For OfflineDiffusionRL, we need to pass the model as the first argument
    if Config.diffusion == "models.CCDiffusion":
        # Create OfflineDiffusionRL with the model as the first argument
        diffusion = diffusion_config(model)
    else:
        # Follow train.py pattern: create model first, then diffusion wrapper
        diffusion = diffusion_config(model)

    # Create trainer
    trainer = trainer_config(diffusion, dataset, renderer)

    # Ensure model is on correct device
    diffusion = diffusion.to(Config.device)

    # Offline training only - no environment setup needed

    if Config.eval_freq > 0:
        print("[ INFO ] Using VAE Diffusion Evaluator for VAE+Diffusion RL model")
        evaluator = utils.VAEDiffusionEvaluator(verbose=True)
        evaluator.init(log_dir=logger.prefix)
        trainer.set_evaluator(evaluator)

    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n",
                color="green",
            )
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])
            if hasattr(trainer, 'ema_model'):
                trainer.ema_model.load_state_dict(state_dict["ema"])

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(diffusion)

    logger.print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)
    if Config.q_pretrain_steps > 0:
        trainer.pretrain_q_function(n_train_steps=Config.q_pretrain_steps)
    for i in range(n_epochs):
        logger.print(f"Epoch {i} / {n_epochs} | {logger.prefix}")
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
    
    trainer.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Add CUDA_LAUNCH_BLOCKING for better error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    # Extract constants from nested config structure
    if "constants" in exp_specs:
        config_dict = exp_specs["constants"]
    else:
        config_dict = exp_specs
    
    # Add meta_data fields if they exist
    if "meta_data" in exp_specs:
        config_dict.update(exp_specs["meta_data"])
    
    Config = build_config_from_dict(config_dict)

    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs.get("exp_name", exp_specs.get("meta_data", {}).get("exp_name", "unknown")),
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: diffusion_loss
                      xKey: steps
                    - yKey: vae_loss
                      xKey: steps
                    - yKey: policy_loss
                      xKey: steps
                    - yKey: value_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN) 