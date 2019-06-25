# Data for Grover

This folder contains some generation examples from Grover. You can get everything by running

`gsutil cp -r "gs://grover-models/generation_examples/*" .`

Alternatively, run
```
wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl
wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~test-probs.npy
wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~val-probs.npy
```

This downloads a dataset of news articles from April 2019, along with Grover-Mega generations. I used this setup to measure the Grover discrimination accuracy. These are just from using Nucleus Sampling `p=0.94` because that was found (from a grid search) to be the hardest to detect when Grover-Mega is both the generator as well as the discriminator. I did this separately for each discriminator-generator combo.

It also downloads the predicted machine/human probabilities given by Grover-Mega as a discriminator. 

To compute the accuracy in the way I did for the paper, use [compute_accuracy_script.py].