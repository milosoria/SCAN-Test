from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_TOKEN = 0


class Attention(nn.Module):
    def __init__(self, hidden_size: int, device=torch.device("cpu")):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, device=device)
        self.Ua = nn.Linear(hidden_size, hidden_size, device=device)
        self.Va = nn.Linear(hidden_size, 1, device=device)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        # Performs a batch matrix-matrix product of matrices
        context = torch.bmm(weights, keys)

        return context, weights


class ActionDecoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.1,
        attention: bool = True,
        device=torch.device("cpu"),
    ):
        super(ActionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, device=device)
        if attention:
            self.attention = Attention(hidden_size, device)
            self.lstm = nn.LSTM(
                2 * hidden_size, hidden_size, n_layers, batch_first=True
            ).to(device)
        else:
            self.attention = None
            self.lstm = nn.LSTM(
                hidden_size, hidden_size, n_layers, batch_first=True
            ).to(device)
        self.out = nn.Linear(hidden_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_cell: torch.Tensor,
        max_length: int,
        target_tensor: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(SOS_TOKEN)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            (
                decoder_output,
                decoder_hidden,
                decoder_cell,
                attn_weights,
            ) = self.forward_step(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing (for predictions and eval): use its own predictions as the next input
                # torch.topk: A namedtuple of (values, indices) is returned with the values and
                # indices of the largest k elements of each row of the input tensor in the given dimension dim.
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        attn_weights = None
        if self.attention is not None:
            embedded = self.dropout(self.embedding(input))
            query = hidden.permute(1, 0, 2)
            keys = encoder_outputs
            # FIXME: keys|| encoder_outputs: (batch_size, seq_length, hidden_size)
            #        query|| hidden: ( batch_size, n_layers, hidden_size)
            context, attn_weights = self.attention(query, keys)
            lstm_input = torch.cat((embedded, context), dim=2)
        else:
            output = self.embedding(input)
            lstm_input = F.relu(output)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.out(output)

        return output, hidden, cell, attn_weights if attn_weights is not None else None

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
