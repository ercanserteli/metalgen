import csv
import json
import os.path as op
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from .metal_dataset import MetalAlbumsDataset, MetalAlbumsResnetDataset, pack_collate_images, pack_collate_resnets
from .models import TitleGenerator


def sample(device="cuda", int2char=None):
    gen = TitleGenerator(220, 2048, 1024, use_images=True).to(device)
    gen.load_state_dict(torch.load("models/title-gen.pt"))

    if int2char is None:
        int2char = load_int2char()

    gen.use_images = True
    gen.eval()

    for i in range(13):
        img = Image.open("data/example-{}.{}".format(i, "png" if i < 10 else "jpg"))
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]
        )
        data = transform(img)
        data = data.unsqueeze(0).to(device)
        sampled_ids = gen.sample(data, int2char)
        sampled_ids = sampled_ids[0].cpu().numpy()

        output = "".join([int2char[i] for i in sampled_ids])
        print(output)


def train():
    torch.random.manual_seed(42)
    dataset = MetalAlbumsResnetDataset("D:\\metal_data\\album_id_titles.json", "D:\\metal_data\\album_resnet.npy")

    # int2char = dataset.int2char
    # with open("int2char.json", "w") as f:
    #     json.dump(int2char, f)

    int2char = load_int2char()
    dataset.set_int2char(int2char)

    # hide_size = int(0.98 * len(dataset))
    # small_size = len(dataset) - hide_size
    # working_dataset, _ = torch.utils.data.random_split(dataset, [small_size, hide_size])
    working_dataset = dataset

    train_size = int(0.8 * len(working_dataset))
    test_size = len(working_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(working_dataset, [train_size, test_size])

    lr = 1e-3
    epochs = 10
    device = "cuda"
    minibatch_size = 100

    train_data_loader = DataLoader(train_dataset, minibatch_size, True, collate_fn=pack_collate_resnets)
    test_data_loader = DataLoader(test_dataset, minibatch_size, True, collate_fn=pack_collate_resnets)

    gen = TitleGenerator(dataset.char_count, 2048, 1024).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params = list(gen.decoder.parameters()) + list(gen.encoder.connection_layer.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Train the models
    train_losses = []
    test_losses = []
    min_test_loss = 999999999999
    best_model_dict = None
    total_step = len(train_data_loader)
    mb_index = 0
    for epoch in range(epochs):
        start_time = timer()
        for i, (resnets, titles, lengths) in enumerate(train_data_loader):
            gen.train()
            titles = titles.to(device)
            targets = pack_padded_sequence(titles, lengths, enforce_sorted=False)

            outputs, packed = gen(resnets, titles, lengths)

            # loss = criterion(outputs, torch.max(targets[0].long(), 1)[1])
            loss = criterion(outputs, targets[0])
            gen.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info and save losses
            if mb_index % 30 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'
                      .format(epoch, epochs, i, total_step, loss.item()))

                padded_targets, target_lens = pad_packed_sequence(targets)
                padded_outputs, output_lens = pad_packed_sequence(packed)

                ex_pred = torch.max(padded_outputs[:output_lens[0], 0, :], 1)[1]
                # ex_true = torch.max(padded_targets[:target_lens[0], 0, :], 1)[1]
                ex_true = padded_targets[:target_lens[0], 0]
                true = "".join([dataset.int2char[c] for c in ex_true.tolist()])
                pred = "".join([dataset.int2char[c] for c in ex_pred.tolist()])
                print("True:", true)
                print("Pred:", pred)

                test_loss = test_model(gen, criterion, test_data_loader, int2char, device)

                train_losses.append(loss.item())
                test_losses.append(test_loss)
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    best_model_dict = gen.state_dict()

            mb_index += 1
        end_time = timer()
        print("Epoch finished in {:.3f} secs".format(end_time - start_time))

    torch.save(best_model_dict, op.join("models", 'title-gen.pt'))

    draw_loss_graphs(train_losses, test_losses)
    print("Best training loss:", min(train_losses))
    print("Best test loss:", min_test_loss)


def load_int2char(subfolder=""):
    with open(op.join(subfolder, "int2char.json"), "r") as f:
        return json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})


def test_model(model, criterion, data_loader, int2char, device="cuda"):
    model.eval()
    with torch.no_grad():
        j = 0
        total_test_loss = 0
        for i, (images, titles, lengths) in enumerate(data_loader):
            images = images.to(device)
            titles = titles.to(device)
            targets = pack_padded_sequence(titles, lengths, enforce_sorted=False)
            outputs, packed = model(images, titles, lengths)
            # loss = criterion(outputs, torch.max(targets[0].long(), 1)[1])
            loss = criterion(outputs, targets[0])
            total_test_loss += loss
            j += 1
        total_test_loss /= j
        print('Test Loss: {:.4f}'.format(total_test_loss.item()))

        padded_targets, target_lens = pad_packed_sequence(targets)
        padded_outputs, output_lens = pad_packed_sequence(packed)

        ex_pred = torch.max(padded_outputs[:output_lens[0], 0, :], 1)[1]
        # ex_true = torch.max(padded_targets[:target_lens[0], 0, :], 1)[1]
        ex_true = padded_targets[:target_lens[0], 0]
        true = "".join([int2char[c] for c in ex_true.tolist()])
        pred = "".join([int2char[c] for c in ex_pred.tolist()])
        print("True:", true)
        print("Pred:", pred)
        return total_test_loss.item()


def draw_loss_graphs(train_losses, test_losses):
    x = np.array(range(len(train_losses))) * 30

    # train_losses = np.insert(train_losses, 0, begin_train_loss)
    # test_losses = np.insert(test_losses, 0, begin_test_loss)

    fig, ax = plt.subplots()
    ax.plot(x, train_losses, label="Train loss")
    ax.plot(x, test_losses, label="Test loss")
    ax.legend()

    ax.set_ybound(lower=0)
    ax.set(xlabel='Minibatch steps', ylabel='Loss', title="Loss Plot")
    ax.grid()

    fig.savefig("loss.png", bbox_inches="tight")
    plt.show()


def extract_resnet_features():
    dataset = MetalAlbumsDataset("D:\\metal_data\\album_id_titles.json", "D:\\metal_data\\covers_128")
    data_loader = DataLoader(dataset, 50, False, collate_fn=pack_collate_images)
    resnet = torch.nn.Sequential(*list(models.wide_resnet101_2(pretrained=True).children())[:-1]).to("cuda")
    resnet.eval()

    all_features = np.zeros((len(dataset), 2048))
    with open('album_resnet.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        index = 0
        for i, (images, ids) in enumerate(data_loader):
            images = images.to("cuda")
            with torch.no_grad():
                features = resnet(images).cpu().numpy()
            all_features[index:index + len(ids), :] = features.reshape(len(ids), 2048)
            index += len(ids)
            for j, album_id in enumerate(ids):
                writer.writerow([album_id, features[j, :, :].reshape(-1).tolist()])
    np.save("album_resnet.npy", all_features)


if __name__ == "__main__":
    # extract_resnet_features()
    # train()
    sample()
