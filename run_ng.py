#!curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
#!ngrok authtoken '2LDDJfaaPPMEOsuPohRhWFRoENU_5LKZYCD4WPDM9mERyBQtG'
import os
import sys
import requests 
from jax.config import config

#from jax.config import config
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = os.environ["TPU_NAME"]

colab_tpu_addr = os.environ["COLAB_TPU_ADDR"].split(":")[0]
url = f"http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607"
requests.post(url)

# The following is required to use TPU Driver as JAX"s backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ["COLAB_TPU_ADDR"]
print(config.FLAGS.jax_backend_target)

import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don"t need them for inference)
params["optimizer"] = optax.scale(0)

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ("dp", "mp")))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")


total_batch = per_replica_batch * jax.device_count() // cores_per_replica
print("stage1")
network = CausalTransformer(params)
print("stage2")
network.state = read_ckpt_lowmem(network.state, "./step_383500/", devices.shape[1])

network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
print("stage3")
# allow text wrapping in generated output: https://stackoverflow.com/a/61401455
from IPython.display import HTML, display

def set_css():
  display(HTML("""
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  """))

def infer(context, top_p=0.9, temp=0.8, gen_len=768):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(tokenizer.decode(o))

    print(f"completion done in {time.time() - start:06}s")
    return samples

print("\"RUN RUN RUN\"")

top_p = 0.95 #@param {type:"slider", min:0, max:1, step:0.1}
temp = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}

context = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."""

print(infer(top_p=top_p, temp=temp, gen_len=512, context=context)[0])
print('\"Check Gen Text\"')


import time

import subprocess
tp_proc = subprocess.Popen(["ngrok", "http", "5000"])

time.sleep(10)

import json
from urllib.request import urlopen
url = "http://127.0.0.1:4040/api/tunnels"
j = json.loads(urlopen(url).read().decode("utf-8"))
print('* Running on ' + j['tunnels'][0]['public_url'])

from flask_cloudflared import run_with_cloudflared
from flask import Flask, redirect, url_for, request
import base64

app = Flask(__name__)

@app.route("/")
def hello():
    return "Working"

@app.route("/infer1", methods=["POST"])
def fl_infer2():
    return request.form["contexts"]


@app.route("/infer", methods=["POST"])
def fl_infer():
    if request.method == "POST":
        all = ""
        try:
            top_p = float(request.form["top_p"])
            temp = float(request.form["temp"])
            gen_len = int(request.form["gen_len"])
            check_rnd_num = request.form["check_rnd_num"]
            all = all + "start_gen\"" + check_rnd_num + "\""
            contexts = request.form["contexts"].split("{||}")
            for idx, context in enumerate(contexts):
                all = all + "GENERATION" + str(idx) + "{|{" + infer(top_p=top_p, temp=temp, gen_len=gen_len, context=base64.b64decode(context).decode("UTF-8"))[0] + "}|}"

        except Exception as e:
            return "An exception occurred: " + repr(e)

        all = all + "check_gen_string\"" + check_rnd_num + "\""

        return all

print("run")
app.run(threaded=False)
