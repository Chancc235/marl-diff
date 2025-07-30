import torch
import numpy as np
from ml_logger import logger
from .training import Trainer


class CCRLTrainer(Trainer):
    """
    Trainer for offline reinforcement learning using diffusion models
    This trainer only uses static datasets without environment interaction
    """
    
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer=None,
        target_update_freq=100,    # Target network update frequency
        **kwargs
    ):
        super().__init__(diffusion_model, dataset, renderer, **kwargs)

        self.target_update_freq = target_update_freq
        self.target_update_tau = kwargs.get('target_update_tau', 0.005)
        # Separate optimizers for different components
        if hasattr(self.model, 'q_function'):
            self.q_optimizer = torch.optim.Adam(
                [param for qf in self.model.q_function for param in qf.parameters()],
                lr=self.q_lr
            )
            self.pattern_q_optimizer = torch.optim.Adam(
                self.model.pattern_q_function.parameters(), 
                lr=self.q_pattern_lr
            )
        else:
            self.q_optimizer = None
            self.pattern_q_optimizer = None

    def pretrain_q_function(self, n_train_steps):
        """Pretrain Q-function"""
        # Ensure model is in training mode
        self.model.train()
        
        for step_idx in range(n_train_steps):
            warmup_batch = next(self.dataloader)
            warmup_batch = self.batch_to_device(warmup_batch)
            q_loss, _ = self.model.compute_q_loss(**warmup_batch)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            if step_idx % 100 == 0:
                logger.print(f"Q pretrain step {step_idx}/{n_train_steps}, Q loss: {q_loss.item():.4f}")
            if step_idx % self.target_update_freq == 0:
                tau = self.target_update_tau
                for i, target_q_function in enumerate(self.model.target_q_functions):
                    for param, target_param in zip(self.model.q_function[i].parameters(), target_q_function.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        logger.print("Q-function pretraining completed")

    def train(self, n_train_steps):
        """Training loop for offline RL"""
        # Ensure model is in training mode
        self.model.train()
        
        # Initialize epoch tracking
        epoch_losses = []
        epoch_metrics = {}
        
        for step_idx in range(n_train_steps):
            # Initialize for this training step
            step_losses = []  # Store loss for each gradient accumulation step
            step_infos = {}   # Store all metrics for this step
            # Q-function warm-up for the first step
            if step_idx == 0 and self.q_optimizer is not None:
                # Perform Q-function warm-up iterations
                q_warmup_steps = n_train_steps // 2
                for warmup_i in range(q_warmup_steps):
                    warmup_batch = next(self.dataloader)
                    warmup_batch = self.batch_to_device(warmup_batch)
                    
                    # Only train Q-function during warm-up
                    pattern_q_loss, _ = self.model.compute_pattern_q_loss(**warmup_batch)
                    self.pattern_q_optimizer.zero_grad()
                    pattern_q_loss.backward()
                    self.pattern_q_optimizer.step()
                    
                    if warmup_i % 100 == 0:
                        logger.print(f"Pattern Q warm-up step {warmup_i}/{q_warmup_steps}, Q loss: {pattern_q_loss.item():.4f}")
                
                logger.print(f"Pattern Q-function warm-up completed after {q_warmup_steps} steps")
                self.pattern_q_optimizer.zero_grad()
            scale_loss_sum = 0    
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = self.batch_to_device(batch)
                
                # Check if model supports decentralized training
                if hasattr(self.model, 'decentralized') and self.model.decentralized:
                    # Decentralized training: iterate over all agents for individual losses
                    agent_loss_tensors = []
                    batch_infos = {}  # Info for this batch

                    # Compute loss for specific agent (without Q-learning and policy optimization)
                    agent_loss, agent_info = self.model.loss(**batch)
                    scale_agent_loss = agent_loss / self.gradient_accumulate_every
                    scale_loss_sum += scale_agent_loss
                    
                    batch_infos = agent_info
                    batch_total_loss = agent_loss

                    
                    # Compute Q-learning and policy losses for all agents (centralized)
                    pattern_q_loss, policy_loss, q_p_info = self.model.compute_q_and_policy_losses(**batch)
                    scale_q_loss = pattern_q_loss / self.gradient_accumulate_every
                    scale_policy_loss = policy_loss / self.gradient_accumulate_every
                    scale_loss_sum += scale_policy_loss

                    # Total loss for this batch
                    batch_total_loss += policy_loss
                    batch_q_loss = pattern_q_loss
                    # Accumulate Q and policy infos
                    for key, val in q_p_info.items():
                        if key not in batch_infos:
                            batch_infos[key] = []
                        batch_infos[key].append(val)
                    
                    # Store loss for this batch
                    step_losses.append(batch_total_loss.item())
                    
                    # Average batch infos across agents and accumulate for this step
                    for key, val_list in batch_infos.items():
                        if key not in step_infos:
                            step_infos[key] = []
                        avg_val = np.mean(val_list)
                        
                        step_infos[key].append(avg_val)
            scale_loss_sum.backward()
            # Update main model parameters
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.pattern_q_optimizer.step()
            self.optimizer.zero_grad()
            self.pattern_q_optimizer.zero_grad()
            # Calculate final metrics for this step
            step_loss = np.mean(step_losses)  # Average loss across gradient accumulation steps
            
            # Average all accumulated infos across gradient accumulation steps
            final_infos = {}
            for key, val_list in step_infos.items():
                final_infos[key] = np.mean(val_list)
            
            # Add total loss to final infos
            final_infos['total_loss'] = step_loss
            
            # Collect epoch data
            epoch_losses.append(step_loss)
            for key, val in final_infos.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(val)
            
            # EMA update
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Checkpointing
            if self.step % self.save_freq == 0:
                self.save()

            # Evaluation
            if self.eval_freq > 0 and self.step % self.eval_freq == 0:
                logger.print(f"[ OfflineRLTrainer ] Running evaluation at step {self.step}", color="green")
                if self.evaluator is not None:
                    # Set model to evaluation mode before evaluation
                    self.model.eval()
                    if hasattr(self, 'ema_model'):
                        self.ema_model.eval()
                    
                    self.evaluate()
                    
                    # Set model back to training mode after evaluation
                    self.model.train()
                    
                    logger.print(f"[ OfflineRLTrainer ] Evaluation completed at step {self.step}", color="green")
                else:
                    logger.print(f"[ OfflineRLTrainer ] Warning: evaluator is None, skipping evaluation", color="yellow")

            # Logging
            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in final_infos.items()]
                )
                
                full_log = f"{self.step}: {step_loss:8.4f} | {infos_str}"
                    
                logger.print(full_log)
                
                # Log metrics
                metrics = {k: v.detach().item() if torch.is_tensor(v) else v for k, v in final_infos.items()}
                
                logger.log(
                    step=self.step, 
                    loss=step_loss, 
                    **metrics,
                    flush=True
                )

            # Sampling and visualization
            if self.sample_freq and self.step % self.sample_freq == 0:
                # Set model to evaluation mode for sampling
                self.model.eval()
                if hasattr(self, 'ema_model'):
                    self.ema_model.eval()
                
                self.render_samples()
                
                # Set model back to training mode
                self.model.train()

            self.step += 1
            
        # Save epoch statistics to CSV
        self._save_epoch_stats_to_csv(epoch_losses, epoch_metrics, n_train_steps)
            
    def _save_epoch_stats_to_csv(self, epoch_losses, epoch_metrics, n_train_steps):
        """Save epoch statistics to CSV file"""
        import csv
        import os
        
        # # Calculate averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        # Create CSV data
        csv_data = {
            'epoch': self.step // n_train_steps,
            **avg_metrics
        }
        
        # CSV file path
        csv_path = os.path.join(self.bucket, logger.prefix, 'epoch_stats.csv')
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_path)
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
            
            # Write header if this is a new file
            if not file_exists:
                writer.writeheader()
            
            # Write data
            writer.writerow(csv_data)
        
        logger.print(f"Epoch statistics saved to {csv_path}")
        logger.print(f"Epoch {csv_data['epoch']}: Average loss = {avg_loss:.4f}")
    
    def batch_to_device(self, batch):
        """Convert batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            return batch.to(self.device)
    