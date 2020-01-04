from random import random

import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def init_weights():
    def init_weights_(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
        elif type(m) == torch.nn.LSTM:
            for name, param in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_normal_(param)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.uniform_(m.weight, -1.0, 1.0)

    return init_weights_


# Partially adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
class Encoder(torch.nn.Module):
    def __init__(self, embed_size, use_images):
        super(Encoder, self).__init__()
        self.use_images = use_images

        resnet = models.wide_resnet101_2(pretrained=True)
        self.connection_layer = torch.nn.Sequential(
            torch.nn.Linear(resnet.fc.in_features, embed_size),
            torch.nn.BatchNorm1d(embed_size),
            torch.nn.Dropout(0.5)
        )
        self.connection_layer.apply(init_weights())

        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, X):
        # If use_images, X is an image. Else, it is resnet features.
        if self.use_images:
            with torch.no_grad():
                features = self.resnet(X)
        else:
            features = X
        features = features.reshape(features.size(0), -1)
        features = self.connection_layer(features)
        return features


class Decoder(torch.nn.Module):
    def __init__(self, chars_size, hidden_size, embed_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(chars_size + 2, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=False, num_layers=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, chars_size + 2),
        )
        self.apply(init_weights())

    def forward(self, features, titles, lengths):
        title_embeddings = self.embedding(titles)
        embeddings = torch.cat((features.unsqueeze(0), title_embeddings), 0)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.output(hiddens[0])

        return outputs, PackedSequence(outputs, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)

    def sample(self, features, int2char, states=None):
        with torch.no_grad():
            sampled_ids = []
            inputs = features.unsqueeze(0)
            max_seg_length = 50
            for i in range(max_seg_length):
                hiddens, states = self.lstm(inputs, states)
                outputs = self.output(hiddens.squeeze(0))
                _, predicted = outputs.max(1)
                if predicted.item() == 1:
                    break
                if (i == 0 or int2char[sampled_ids[-1].item()] == " ") and random() < 0.5:
                    outputs[:, predicted] = -9999
                    _, predicted = outputs.max(1)
                    if predicted.item() == 1:
                        break
                sampled_ids.append(predicted)
                inputs = self.embedding(predicted).unsqueeze(0)
            sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class TitleGenerator(torch.nn.Module):
    def __init__(self, vocab_size, decoder_hidden_size, embed_size, use_images=False):
        super(TitleGenerator, self).__init__()
        self.encoder = Encoder(embed_size, use_images)
        self.decoder = Decoder(vocab_size, decoder_hidden_size, embed_size)

    def forward(self, images, titles, lengths):
        return self.decoder(self.encoder(images), titles, lengths)

    def sample(self, images, int2char):
        self.eval()
        return self.decoder.sample(self.encoder(images), int2char)