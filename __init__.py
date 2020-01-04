import base64
import json
import os
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_basicauth import BasicAuth

from generators import MetalGenerator


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='hebebebe', TEMPLATES_AUTO_RELOAD=True, BASIC_AUTH_USERNAME="hede",
                            BASIC_AUTH_PASSWORD="hododo")

    basic_auth = BasicAuth(app)
    os.makedirs(app.instance_path, exist_ok=True)

    gen = MetalGenerator()

    @app.route('/')
    @basic_auth.required
    def homepage():
        albums = []
        for i in range(12):
            cover, title, latents = gen.generate_cover_and_title()
            output = BytesIO()
            cover.save(output, format='PNG')
            output.seek(0)
            output_s = output.read()
            b64 = base64.b64encode(output_s)
            albums.append((b64.decode("utf-8"), title, latents.squeeze().tolist()))
        return render_template("index.html", albums=albums)

    @app.route('/style-mix', methods=['POST'])
    def style_mix():
        data = request.json
        latent_1 = np.array(json.loads(data.get("latent_1")))
        latent_2 = np.array(json.loads(data.get("latent_2")))

        image, title = gen.style_mix(latent_1, latent_2)
        output = BytesIO()
        image.save(output, format='PNG')
        output.seek(0)
        output_s = output.read()
        b64 = base64.b64encode(output_s)
        return jsonify(cover=b64.decode("utf-8"), title=title)

    return app


