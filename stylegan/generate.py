import os
import pickle
import numpy as np
import PIL.Image
from dnnlib import tflib
import config


def main():
    tflib.init_tf()
    _G, D, Gs = pickle.load(open("network-final.pkl", "rb"))
    # Gs.print_layers()

    for i in range(10):
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example-'+str(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
