import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class AdvancePolicyNet(nn.Module):
    """
    策略和 Q_value 网络
    策略网络的输出为 n_actions
    Q_value 网络的输出为 1
    """

    def __init__(self, n_input, n_output, is_value=False):
        super(AdvancePolicyNet, self).__init__()

        self.is_value = is_value

        self.common = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

        self.policy = nn.Sequential(
            nn.Softmax(dim=1)
        )

        self.apply(init_weight)

    def forward(self, x):
        x = self.common(x)

        if not self.is_value:
            x = self.policy(x)

        return x
