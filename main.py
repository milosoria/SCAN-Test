# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# LSTM, attention, one layer, 100 hidden units, dropout 0.1
import torch
from torch import nn, optim
import random
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from lang import Lang
from typing import Tuple
import numpy as np
from encoder import CommandEncoder
from decoder import ActionDecoder
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker
import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1
PATH = "../SCAN"
# Get from paper

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device}")
device = torch.device(device)


def get_max_length(data: list[Tuple[str, str]]) -> int:
    max = 0
    for input, output in data:
        compare_len = len(input) if len(input) > len(output) else len(output)
        if compare_len > max:
            max = compare_len
    return max


def read_file(relative_path: str) -> list[str]:
    with open(f"{PATH}/{relative_path}", "r") as f:
        data = f.readlines()
    return data


def preprocess(
    data: list[str],
    input_lang_name: str,
    output_lang_name: str,
) -> Tuple[Lang, Lang, list[Tuple[str, str]]]:
    pairs = []
    print("Read %i %s-%s lines" % (len(data), input_lang_name, output_lang_name))
    for line in data:
        primitives, commands = line[4:].split(" OUT: ")
        commands = commands.strip("\n")
        pairs.append((primitives, commands))
    return Lang(input_lang_name), Lang(output_lang_name), pairs


def load_langs(
    input_lang_name: str, output_lang_name: str, data: list[str]
) -> Tuple[Lang, Lang, list[Tuple[str, str]]]:
    input_lang, output_lang, pairs = preprocess(data, input_lang_name, output_lang_name)
    for input, output in pairs:
        input_lang.add_sentence(input)
        output_lang.add_sentence(output)
    print("Counted Words")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def word_to_index(lang: Lang, sentence: str) -> list[int]:
    return [lang.word2index[word] for word in sentence.split(" ")]


def sentence_to_tensor(lang: Lang, sentence: str) -> torch.Tensor:
    indexes = word_to_index(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def pair_to_tensor(pair: Tuple[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensor = sentence_to_tensor(input_lang, pair[0])
    target_tensor = sentence_to_tensor(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(
    batch_size, data
) -> Tuple[Lang, Lang, DataLoader, int, list[Tuple[str, str]]]:
    input_lang, output_lang, pairs = load_langs("primitives", "commands", data)
    max_length = get_max_length(pairs)
    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    target_ids = np.zeros((n, max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = word_to_index(input_lang, inp)
        tgt_ids = word_to_index(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True
    )
    return input_lang, output_lang, train_dataloader, max_length, pairs


def train_epoch(
    dataloader: DataLoader,
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
) -> float:
    # Training Loop
    total_loss = 0
    encoder_hidden, encoder_cell = (None, None)
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, (encoder_hidden, encoder_cell) = encoder(
            input_tensor,
            encoder_hidden.detach() if encoder_hidden is not None else None,
            encoder_cell.detach() if encoder_cell is not None else None,
        )
        decoder_outputs, _, _ = decoder(
            encoder_outputs, encoder_hidden, encoder_cell, max_length, target_tensor
        )

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def show_plot(points: list[float]):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    sentence: str,
    input_lang: Lang,
    output_lang: Lang,
) -> Tuple[list[str], torch.Tensor]:
    encoder_hidden, encoder_cell = (None, None)
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_lang, sentence)

        encoder_outputs, (encoder_hidden, encoder_cell) = encoder(
            input_tensor, encoder_hidden, encoder_cell
        )
        decoder_outputs, _, decoder_attn = decoder(
            encoder_outputs, encoder_hidden, encoder_cell, max_length, None
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_randomly(
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    pairs: list[Tuple[str, str]],
    n: int = 10,
):
    for _ in range(n):
        pair = random.choice(pairs)
        print("INPUT: ", pair[0])
        print("TARGET: ", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print("OUTPUT: ", output_sentence)
        print("")


if __name__ == "__main__":
    hparams = {
        "batch_size": 32,
        "hidden_size": 100,
        "n_epochs": 10000,
        "n_layers": 1,
        "lr": 0.001,
        "dropout": 0.1,
        "print_every": 10,
        "plot_every": 100,
        "save_every": 100,
        "eval_every": 100,
    }

    train_data = read_file("length_split/tasks_train_length.txt")
    test_data = read_file("length_split/tasks_test_length.txt")

    input_lang, output_lang, train_dataloader, max_length, pairs = get_dataloader(
        hparams["batch_size"], train_data
    )

    _, _, test_dataloader, _, test_pairs = get_dataloader(
        hparams["batch_size"], test_data
    )

    encoder = CommandEncoder(
        input_size=input_lang.n_words,
        hidden_size=hparams["hidden_size"],
        n_layers=hparams["n_layers"],
        dropout=hparams["dropout"],
        device=device,
    )
    encoder.load("models/encoder_200.pt")
    decoder = ActionDecoder(
        output_size=output_lang.n_words,
        hidden_size=hparams["hidden_size"],
        n_layers=hparams["n_layers"],
        dropout=hparams["dropout"],
        attention=True,
        device=device,
    )
    decoder.load("models/decoder_200.pt")
    criterion = torch.nn.CrossEntropyLoss()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hparams["lr"])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hparams["lr"])
    criterion = nn.NLLLoss()

    for epoch in range(1, hparams["n_epochs"] + 1):
        print("Epoch: ", epoch)
        encoder.train()
        decoder.train()
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % hparams["print_every"] == 0:
            print_loss_avg = print_loss_total / hparams["print_every"]
            print_loss_total = 0
            print(
                "(%d %d%%) %.4f"
                % (
                    epoch,
                    epoch / hparams["n_epochs"] * 100,
                    print_loss_avg,
                )
            )

        if epoch % hparams["plot_every"] == 0:
            plot_loss_avg = plot_loss_total / hparams["plot_every"]
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            if len(plot_losses) > 100:
                show_plot(plot_losses)
        if epoch % hparams["save_every"] == 0:
            encoder.save(f"models/encoder_{epoch}.pt")
            decoder.save(f"models/decoder_{epoch}.pt")

        if epoch % hparams["eval_every"] == 0:
            encoder.eval()
            decoder.eval()
            evaluate_randomly(encoder, decoder, test_pairs)
