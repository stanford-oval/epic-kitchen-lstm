import pytorch_lightning
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, network_type: str, input_size, output_size, hidden_size, dropout, num_layers):
        super().__init__()
        if network_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True)
        elif network_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=2,
                              dropout=0.1,
                              batch_first=True)
        self.fc = nn.Linear(in_features=20, out_features=output_size)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        return self.fc(r_out[:, -1, :])


class ActionPredictor(pytorch_lightning.core.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.network = SimpleLSTM(
            cfg.MODEL.LSTM_TYPE,
            cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1],
            cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1],
            20,
            0.1,
            cfg.MODEL.LSTM_LAYER_NUM)

    def forward(self, x):
        return self.network(x)


class ActionPredictorNoPred(pytorch_lightning.core.LightningModule):
    def __init__(self, cfg, output_dim):
        super().__init__()
        self.network = SimpleLSTM(
            cfg.MODEL.LSTM_TYPE,
            cfg.MODEL.NUM_CLASSES[0] + cfg.MODEL.NUM_CLASSES[1],
            output_dim,
            20,
            0.1,
            cfg.MODEL.LSTM_LAYER_NUM)

    def forward(self, x):
        r_out, _ = self.network.rnn(x, None)
        return r_out[:, -1, :]
