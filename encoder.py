from typing import Tuple
import torch
import torch.nn as nn


class CommandEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 100,
        n_layers: int = 1,
        dropout: float = 0.1,
        device=torch.device("cpu"),
    ):
        super(CommandEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, device=device)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        ).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch_size, seq_length)?
        embeds = self.dropout(self.embedding(x))
        # embeds: (batch_size, seq_length, hidden_size)
        if hidden is None or cell is None:
            output, (hidden, cell) = self.lstm(embeds)
        else:
            output, (hidden, cell) = self.lstm(embeds, (hidden, cell))
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (n_layers, batch_size, hidden_size)
        # cell: (n_layers, batch_size, hidden_size)
        return output, (hidden, cell)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
