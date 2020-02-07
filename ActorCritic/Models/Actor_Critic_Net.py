import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class ActorCriticNet(nn.Module):
    """
    Actor 网络与 Critic 网络共享参数
    """

    def __init__(self, n_input, n_output):
        super(ActorCriticNet, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, n_output),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.apply(init_weight)

    def forward(self, x):
        x = self.common(x)
        act_prob = self.actor(x)
        value = self.critic(x)
        return act_prob, value
