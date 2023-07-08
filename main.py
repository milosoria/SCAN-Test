from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from typing import Tuple
from lang import Lang
from encoder import CommandEncoder
import random
from decoder import ActionDecoder
from utils import (
    device,
    get_dataloader,
    show_plot,
    log_it,
    get_max_length,
    load_langs,
    read_file,
    sentence_to_tensor,
    EOS_TOKEN,
    epoch_loop,
)


def evaluate(
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    pair: Tuple[str, str],
    input_lang: Lang,
    output_lang: Lang,
) -> Tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder_hidden, encoder_cell = (None, None)
    (input_sentence, input_target) = pair
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_lang, input_sentence)
        target_tensor = sentence_to_tensor(output_lang, input_target)
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

        target_tensor = F.pad(
            target_tensor,
            (0, max_length - target_tensor.size(1)),
            "constant",
            EOS_TOKEN,
        )
        topi = topi.squeeze()
        acc = torch.sum((topi == target_tensor)) / len(target_tensor)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1),
        )
    return decoded_words, decoder_attn, loss, acc


def evaluate_randomly(
    encoder: CommandEncoder,
    decoder: ActionDecoder,
    input_lang: Lang,
    output_lang: Lang,
    pairs: list[Tuple[str, str]],
    n: int = 10,
):
    for _ in range(n):
        pair = random.choice(pairs)
        output_words, _, loss, acc = evaluate(
            encoder=encoder,
            decoder=decoder,
            pair=pair,
            input_lang=input_lang,
            output_lang=output_lang,
        )
        output_sentence = " ".join(output_words)
        output_str = f"INPUT: {pair[0]}\nTARGET: {pair[1]}\nOUTPUT: {output_sentence}\nLOSS:\
                {loss}\n ACC: {acc}\n"
        log_it(output_str, experiment)


def train_or_test(test: bool = False):
    plot_losses = []
    plot_accs = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0  # Reset every print_every
    plot_acc_total = 0  # Reset every plot_every
    for epoch in range(1, hparams["n_epochs"] + 1):
        if test:
            encoder.eval()
            decoder.eval()
        else:
            encoder.train()
            decoder.train()
        loss, acc = epoch_loop(
            test_dataloader if test else train_dataloader,
            encoder,
            decoder,
            max_length,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_acc_total += acc
        plot_acc_total += acc
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % hparams["print_every"] == 0:
            print_loss_avg = print_loss_total / hparams["print_every"]
            print_acc_avg = print_acc_total / hparams["print_every"]
            print_loss_total = 0
            print_acc_total = 0
            output_str = "(Epoch: %d, Progress: %d%%) Loss: %.4f Acc: %.4f" % (
                epoch,
                epoch / hparams["n_epochs"] * 100,
                print_loss_avg,
                print_acc_avg,
            )
            log_it(output_str, experiment)

        if epoch % hparams["plot_every"] == 0:
            plot_loss_avg = plot_loss_total / hparams["plot_every"]
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            if len(plot_losses) >= 100:
                show_plot(plot_losses)
            plot_acc_avg = plot_acc_total / hparams["plot_every"]
            plot_accs.append(plot_acc_avg)
            plot_acc_total = 0
            if len(plot_accs) >= 100:
                show_plot(plot_accs)

        if epoch % hparams["save_every"] == 0:
            if not test:
                encoder.save(f"models/{experiment}/train/encoder_{epoch}.pt")
                decoder.save(f"models/{experiment}/train/decoder_{epoch}.pt")
            np.save(
                f"models/{experiment}/{'test' if test else 'train'}/plot_losses_{epoch}.npy",
                plot_losses,
            )
            np.save(
                f"models/{experiment}/{'test' if test else 'train'}/plot_accs_{epoch}.npy",
                plot_accs,
            )

        if epoch % hparams["eval_every"] == 0:
            print("Evaluating: ")
            evaluate_randomly(
                encoder=encoder,
                decoder=decoder,
                input_lang=input_lang,
                output_lang=output_lang,
                pairs=test_pairs,
            )


if __name__ == "__main__":
    hparams = {
        "batch_size": 32,
        "hidden_size": 100,
        "n_epochs": 500,
        "n_layers": 1,
        "lr": 0.001,
        "dropout": 0.1,
        "print_every": 10,
        "plot_every": 100,
        "save_every": 100,
        "eval_every": 100,
    }
    experiment = "length_split"
    train_data = read_file(f"{experiment}/tasks_train_length.txt")
    test_data = read_file(f"{experiment}/tasks_test_length.txt")
    # load langs with train and test data, so both have all the words from their respective domains
    input_lang, output_lang, train_pairs, test_pairs = load_langs(
        input_lang_name="primitives",
        output_lang_name="commands",
        train_data=train_data,
        test_data=test_data,
    )

    max_length = get_max_length(train_pairs)

    (input_lang, output_lang, train_dataloader, train_pairs) = get_dataloader(
        batch_size=hparams["batch_size"],
        max_length=max_length,
        input_lang=input_lang,
        output_lang=output_lang,
        pairs=train_pairs,
    )

    (input_lang, output_lang, test_dataloader, test_pairs) = get_dataloader(
        batch_size=hparams["batch_size"],
        max_length=max_length,
        input_lang=input_lang,
        output_lang=output_lang,
        pairs=test_pairs,
    )

    encoder = CommandEncoder(
        input_size=input_lang.n_words,
        hidden_size=hparams["hidden_size"],
        n_layers=hparams["n_layers"],
        dropout=hparams["dropout"],
        device=device,
    )
    encoder.load(f"models/{experiment}/encoder_500.pt")
    decoder = ActionDecoder(
        output_size=output_lang.n_words,
        hidden_size=hparams["hidden_size"],
        n_layers=hparams["n_layers"],
        dropout=hparams["dropout"],
        attention=True,
        device=device,
    )
    decoder.load(f"models/{experiment}/decoder_500.pt")

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hparams["lr"])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hparams["lr"])
    criterion = nn.NLLLoss()
    train_or_test()
    train_or_test(True)
