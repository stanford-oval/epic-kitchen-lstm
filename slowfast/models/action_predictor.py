import pytorch_lightning
from torch import nn


class ActionPredictor(pytorch_lightning.core.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.LSTM(input_size=cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1],
                           hidden_size=20,
                           num_layers=2,
                           dropout=0.1,
                           batch_first=True)
        self.fc = nn.Linear(in_features=20, out_features=cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1])

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        return self.fc(r_out[:, -1, :])


class ActionPredictorNoPred(pytorch_lightning.core.LightningModule):
    def __init__(self, cfg, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1],
                           hidden_size=output_dim,
                           num_layers=2,
                           dropout=0.1,
                           batch_first=True)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        return r_out[:, -1, :]
