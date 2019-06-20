# Grover
(aka, code for [Defending Against Neural Fake News](https://arxiv.org/abs/1905.12616))

Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. 

Visit our project page at [rowanzellers.com/grover](https://rowanzellers.com/grover), [the AI2 online demo](https://grover.allenai.org), or read the full paper at [arxiv.org/abs/1905.12616](https://arxiv.org/abs/1905.12616). 

![teaser](https://i.imgur.com/VAGFpBe.png "teaser")

## What's in this repo?

We are releasing the following:
* Code for the Grover generator (in [lm/](lm/)). This involves training the model as a language model across fields.
* Code for the Grover discriminator in [discrimination/](discrimination/). Without much changing, you can run Grover as a discriminator to detect Neural Fake News.
* Code for generating from a Grover model, in [sample/](sample/).
* Code for making your own RealNews dataset in [realnews/](realnews/).
* Model checkpoints freely available online for the Grover-Base and Grover-Large models. For Grover-Mega or the RealNews dataset, please [submit this form](https://docs.google.com/forms/d/1LMAUeUtHNPXO9koyAIlDpvyKsLSYlrBj3rYhC30a7Ak) and we will get back to you as soon as possible.

Scroll down 👇 for some easy-to-use instructions for setting up Grover to generate news articles.

## Setting up your environment

*NOTE*: If you just care about making your own RealNews dataset, you will need to set up your environment separately just for that, using an AWS machine (see [realnews/](realnews/).)

There are a few ways you can run Grover:
* **Generation mode (inference)**. This requires a GPU because I wasn't able to get top-p sampling, or caching of transformer hidden states, to work on a TPU.
* **LM Validation mode (perplexity)**. This could be run on a GPU or a TPU, but I've only tested this with TPU inference.
* **LM Training mode**. This requires a large TPU pod.
* **Discrimination mode (training)**. This requires a TPU pod.
* **Discrimination mode (inference)**. This could be run on a GPU or a TPU, but I've only tested this with TPU inference.

I used Python3.6 for everything. Usually I set it up using the following commands:
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/conda && \
     rm ~/miniconda.sh && \
     ~/conda/bin/conda install -y python=3.6
```
Then `pip install -r requirements-gpu.txt` if you're installing on a GPU, or `pip install requirements-tpu.txt` for TPU.

Misc notes/tips:
* If you have a lot of projects on your machine, you might want to use an anaconda environment to handle them all. Use `conda create -n grover python=3.6` to create an environment named `grover`. To enter the environment use `source activate grover`. To leave use `source deactivate`.
* I'm using tensorflow `1.13.1` which requires Cuda `10.0`. You'll need to install that from the nvidia website. I usually install it into `/usr/local/cuda-10.0/`, so you will need to run `export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64` so tensorflow knows where to find it. 
* I always have my pythonpath as the root directory. While in the `grover` directory, run `export PYTHONPATH=$(pwd)` to set it.

## Quickstart: setting up Grover for generation!

1. Set up your environment. Here's the easy way, assuming anaconda is installed: `conda create -y -n grover python=3.6 && source activate grover && pip install -r requirements-gpu.txt`
2. Download the model using `python download_model.py base`
3. Now generate: `PYTHONPATH=$(pwd) python sample/contextual_generate.py -model_config_fn lm/configs/base.json -model_ckpt models/base/model.ckpt -metadata_fn sample/april2019_set_mini.jsonl -out_fn april2019_set_mini_out.jsonl`

Congrats! You can view the generations, conditioned on the domain/headline/date/authors, in `april2019_set_mini_out.jsonl`.


### Bibtex

```
@inproceedings{zellers2019grover,
    title={Defending Against Neural Fake News},
    author={Zellers, Rowan and Holtzman, Ari and Rashkin, Hannah and Bisk, Yonatan and Farhadi, Ali and Roesner, Franziska and Choi, Yejin},
    journal={arXiv preprint arXiv:1905.12616},
    year={2019}
}
```