# What everything does

* `validate.py` gets perplexity. You can use the script `validate.sh` which contains some arguments I used.
* `train.py` trains a grover model from scratch. The script `train_tpu_adafactor.sh` could also help. You probably don't want to do this unless you have a lot of money for TPUs. However, it might be a good idea to finetune Grover to a different domain.

# Setting up tensorboard

During training, you can use tensorboard by running the following commands:

```
ssh -L 6006:localhost:6006 myservername
tensorboard --logdir="grover":"gs://MYOUTPUTPATH" --port=6006
```