#!/usr/bin/env bash

DUMP2CRAWL=$1
aws s3 cp "s3://commoncrawl/crawl-data/${DUMP2CRAWL}/warc.paths.gz" ~/temp/
rm -f ~/temp/warc.paths
gunzip  ~/temp/warc.paths.gz

parallel -j $(nproc --all) --will-cite python process_ccrawl.py -path "{1}" ">" "~/logs/{%}.txt" < ~/temp/warc.paths
