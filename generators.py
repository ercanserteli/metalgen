import torch
import tensorflow as tf

from titlegen.gen_title import load_int2char
from titlegen.models import TitleGenerator

from torchvision import transforms
import sys
import os
import pickle
import numpy as np
import PIL.Image

from stylegan.dnnlib import tflib
from stylegan import dnnlib

# This is needed for pickle to work
sys.modules['dnnlib'] = dnnlib


class MetalGenerator:
    def __init__(self):
        self.tf_session = tflib.init_tf()
        self.tf_graph = tf.get_default_graph()
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        _G, D, Gs = pickle.load(open("stylegan/network-final.pkl", "rb"))
        self.cover_gen = Gs
        self.device = "cuda"
        self.title_gen = TitleGenerator(220, 2048, 1024, use_images=True).to(self.device)
        self.title_gen.load_state_dict(torch.load("titlegen/models/title-gen.pt"))
        self.title_gen.use_images = True
        self.int2char = load_int2char("titlegen")
        self.rnd = np.random.RandomState(None)

        self.transform = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))]
                )

    def generate_cover_and_title(self):
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                latents = self.rnd.randn(1, self.cover_gen.input_shape[1])
                fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                images = self.cover_gen.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # png_filename = os.path.join("", 'cover.png')
        image = PIL.Image.fromarray(images[0], 'RGB')
        # image.save(png_filename)
        data = self.transform(image)
        data = data.unsqueeze(0).to(self.device)
        sampled_ids = self.title_gen.sample(data, self.int2char)
        sampled_ids = sampled_ids[0].cpu().numpy()

        title = "".join([self.int2char[i] for i in sampled_ids])
        # print(title)
        # image.show(title)
        return image, title, latents

    def style_mix(self, latent_1, latent_2):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                # src_seed = 42
                # dst_seed = 82
                style_range = range(8, 12)
                fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                # src_latent = np.expand_dims(np.random.RandomState(src_seed).randn(self.cover_gen.input_shape[1]), 0)
                # dst_latent = np.expand_dims(np.random.RandomState(dst_seed).randn(self.cover_gen.input_shape[1]), 0)
                src_latent = np.expand_dims(latent_1, 0)
                dst_latent = np.expand_dims(latent_2, 0)
                src_dlatent = self.cover_gen.components.mapping.run(src_latent, None)  # [seed, layer, component]
                dst_dlatent = self.cover_gen.components.mapping.run(dst_latent, None)  # [seed, layer, component]
                # src_images = self.cover_gen.components.synthesis.run(src_dlatent, randomize_noise=False, output_transform=fmt)
                # dst_images = self.cover_gen.components.synthesis.run(dst_dlatent, randomize_noise=False, output_transform=fmt)

                row_dlatent = np.expand_dims(dst_dlatent[0], 0)
                row_dlatent[:, style_range] = src_dlatent[:, style_range]
                image = self.cover_gen.components.synthesis.run(row_dlatent, randomize_noise=False, output_transform=fmt)

        image = PIL.Image.fromarray(image[0], 'RGB')
        data = self.transform(image)
        data = data.unsqueeze(0).to(self.device)
        sampled_ids = self.title_gen.sample(data, self.int2char)
        sampled_ids = sampled_ids[0].cpu().numpy()

        title = "".join([self.int2char[i] for i in sampled_ids])
        return image, title
