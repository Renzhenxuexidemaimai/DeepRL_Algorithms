import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, num_states, num_actions):
        super(MLPPolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_states, 50),
            nn.ReLU(),
            nn.Linear(50, num_actions)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        return out
