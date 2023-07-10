import torch
import numpy as np
from typing import Tuple, Union
from lang import Lang
from encoder import CommandEncoder
from decoder import ActionDecoder
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker


PATH = "../SCAN"
EOS_TOKEN = 1
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device = torch.device(device)


def read_file(relative_path: str) -> list[str]:
    with open(f"{PATH}/{relative_path}", "r") as f:
        data = f.readlines()
    return data


def preprocess(
    data: list[str],
) -> list[Tuple[str, str]]:
    pairs = []
    for line in data:
        primitives, commands = line[4:].split(" OUT: ")
        commands = commands.strip("\n")
        pairs.append((primitives, commands))
    return pairs


def load_langs(
    input_lang_name: str,
    output_lang_name: str,
    train_data: list[str],
    test_data: list[str],
) -> Tuple[Lang, Lang, list[Tuple[str, str]], list[Tuple[str, str]]]:
    print(
        "Train Split: Read %i %s-%s lines"
        % (len(train_data), input_lang_name, output_lang_name)
    )
    print(
        "Test Split: Read %i %s-%s lines"
        % (len(test_data), input_lang_name, output_lang_name)
    )

    input_lang, output_lang = Lang(input_lang_name), Lang(output_lang_name)
    train_pairs = preprocess(train_data)
    test_pairs = preprocess(test_data)

    for input, output in train_pairs:
        input_lang.add_sentence(input)
        output_lang.add_sentence(output)

    for input, output in test_pairs:
        input_lang.add_sentence(input)
        output_lang.add_sentence(output)
    print("Counted Words")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, train_pairs, test_pairs


def get_max_length(data: list[Tuple[str, str]]) -> int:
    max = 0
    for input, output in data:
        compare_len = len(input) if len(input) > len(output) else len(output)
        if compare_len > max:
            max = compare_len
    return max


def word_to_index(lang: Lang, sentence: str) -> list[int]:
    return [lang.word2index[word] for word in sentence.split(" ")]


def sentence_to_tensor(lang: Lang, sentence: str) -> torch.Tensor:
    indexes = word_to_index(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(
    batch_size: int,
    max_length: int,
    input_lang: Lang,
    output_lang: Lang,
    pairs: list[Tuple[str, str]],
) -> DataLoader:
    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    target_ids = np.zeros((n, max_length), dtype=np.int32)
    input_ids.fill(EOS_TOKEN)
    target_ids.fill(EOS_TOKEN)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = word_to_index(input_lang, inp)
        tgt_ids = word_to_index(output_lang, tgt)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True
    )
    return train_dataloader


def show_pl∫ot(points: list[float]):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def log_it(output_str: str, experiment: str, train: bool = True):
    print(output_str)
    with open(f"logs/{experiment}/{'train' if train else 'test'}/logs.txt", "a") as f:
        f.write(output_str)


def epoch_loop(
    dataloader: DataLoader,
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    max_length: int,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    testing: bool = False,
) -> Tuple[float, Union[torch.Tensor, float]]:
    # Training Loop
    total_loss = 0
    total_acc = 0
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

        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        target_tensor = target_tensor.view(-1)
        loss = criterion(decoder_outputs, target_tensor)

        _, topi = decoder_outputs.topk(1)
        topi = topi.squeeze()
        acc = (torch.sum((topi == target_tensor)) / len(target_tensor)).cpu().item()
        if not testing:
            loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)
