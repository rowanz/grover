# Downloading the existing RealNews dataset

A tiny version is available for debugging purposes (`realnews_tiny.jsonl`). You can download the full version, just please [submit this form](https://docs.google.com/forms/d/1LMAUeUtHNPXO9koyAIlDpvyKsLSYlrBj3rYhC30a7Ak).

# Code for scraping the realnews dataset

You probably don't want to create your own realnews dataset. But for reproducibility purposes here's what I did:

Setup:
* You need to spawn off an AWS machine in `us-east-1`. That's where [common crawl](https://registry.opendata.aws/commoncrawl/) is located. My recommendation is to get a machine with as many CPUs as possible. We used roughly 15 machines, each with 72 CPUs. Thankfully, common crawl is broken up into many pieces so the work can be easily distributed amongst these machines
* Make a new `s3` bucket, which also needs to be in the `us-east-1` region.


Now, let's get started:
* Use any of the ids in cc_files (like the last one, `CC-MAIN-2019-13` which is for March 2019).
* Then run `process_ccrawl.sh CC-MAIN-2019-13`, and this will crawl that in parallel using all of your CPUs. You will probably need to change the arguments in `process_ccrawl.py` so it goes to your bucket.
* Do this a lot, and now you probably need to deduplicate, so use `dedupe_crawl.py`. This can be done on 1 CPU.
* Last, you'll want to convert everything to tfrecords and move them to Google Cloud. You also probably want to do this in parallel. This can be done using `prepare_lm_data.sh`

That's it!