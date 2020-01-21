import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class AdvancePolicy(nn.Module):
    """
    策略和 Q_value 网络
    策略网络的输出为 n_actions
    Q_value 网络的输出为 1
    """

    def __init__(self, n_state, n_action, n_hidden=128):
        super(AdvancePolicy, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.LeakyReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(n_hidden, n_action),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(n_hidden, 1),
        )

        self.apply(init_weight)

    def forward(self, x):
        return self.common(x)

    def get_action_probs(self, state):
        common = self.forward(state)
        return self.policy(common)

    def get_value(self, state):
        common = self.forward(state)
        return self.value(common)