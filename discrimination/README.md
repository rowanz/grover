# discrimination

This folder contains code for the discrimination experiments.

`run_discrimination.py` can be used to train or evaluate a model for discrimination

# Discrimination checkpoints
Here are links to the discrimination checkpoints. You'll need to use google cloud storage to download these.

**NOTE**: These checkpoints were trained on 5000 examples from a specific Grover generator, with a specific nucleus sampling top-p setting. As a result, these aren't necessarily the best discrimination checkpoints, nor are they the most general. The reason we used this experimental setup is outlined [in the paper](https://arxiv.org/abs/1905.12616) -- we assumed limited access to the generator. We did [later experiments](https://medium.com/ai2-blog/counteracting-neural-disinformation-with-grover-6cf6690d463b) and found that if you assume, say, 100k examples from a generator, you'll do much better (up to around 97% accuracy).

In other words, if you want to mimic my experimental setup, but with your own generator, you'd also need to train your own discriminator from scratch. Alternatively, if you want a really good discriminator against my checkpoints for whatever reason, you'd also probably want to train your own discriminator from scratch.

Medium trained on medium, top-p=0.96:
```
gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.data-00000-of-00001
gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.index
gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.meta
```

Mega trained on mega, top-p=0.94:
```
gs://grover-models/discrimination/generator=mega~discriminator=grover~discsize=mega~dataset=p=0.94/model.ckpt-1562.data-00000-of-00001
gs://grover-models/discrimination/generator=mega~discriminator=grover~discsize=mega~dataset=p=0.94/model.ckpt-1562.index
gs://grover-models/discrimination/generator=mega~discriminator=grover~discsize=mega~dataset=p=0.94/model.ckpt-1562.meta
```