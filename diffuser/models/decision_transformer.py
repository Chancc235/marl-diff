import numpy as np
import torch
import torch.nn as nn
import transformers
from diffuser.models.trajectory_gpt2 import GPT2Model
import torch.nn.functional as F

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            hidden_size=128,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            **kwargs
        )
        self.hidden_size = hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh()
        )
        
        # self.predict_action = nn.Sequential(
        #     nn.Linear(hidden_size, self.act_dim),
        #     nn.Softmax(dim=-1)  # 离散动作，确保输出为概率分布
        # )


    def forward(self, states, actions, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(states.device)

        timesteps = torch.arange(seq_length, device=states.device).unsqueeze(0).repeat(batch_size, 1)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        actions = actions.to(torch.float32)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings.view(time_embeddings.shape[0], time_embeddings.shape[1], time_embeddings.shape[2])
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.view(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)


        action_preds = self.predict_action(x[:,0])  # predict next action given state

        return x, action_preds
    
    def get_embeddings(self, states, actions, attention_mask=None):
        x, _ = self.forward(states, actions, attention_mask=attention_mask)
        return x[:,0,-1]

# =========================
# 测试代码
# =========================
if __name__ == "__main__":
    # 假设状态维度为4，动作维度为2，隐藏层为16，序列长度为5，batch为3
    state_dim = 4
    act_dim = 2
    hidden_size = 16
    seq_len = 5
    batch_size = 3
    n_head = 4
    n_layer = 2

    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, hidden_size=hidden_size, n_head=n_head, n_layer=n_layer, max_ep_len=10, dropout=0.1)
    print(model.transformer.config)

    # 随机生成输入
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, act_dim)

    # 前向测试
    x, action_preds = model(states, actions)
    loss = F.mse_loss(action_preds, actions)
    print("loss:", loss)
    embeddings = model.get_embeddings(states, actions)
    print("x shape:", x.shape)
    print("action_preds shape:", action_preds.shape)
    print("embeddings shape:", embeddings.shape)