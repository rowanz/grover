# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from lm.modeling import model_fn_builder, GroverConfig
import tensorflow as tf
from lm.dataloader import input_fn_builder
import numpy as np
import tempfile
import h5py
from google.cloud import storage

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "config_file", 'configs/base.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "validation_name", 'preds.h5',
    "Name to use")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained model).")

flags.DEFINE_integer(
    "max_seq_length", 1024,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("batch_size", 32, "Batch size used for eval")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


# This is a handy little utility so that we can save the perplexities to TPU
class gcloudwriter():
    def __init__(self, gcloud_name):
        assert gcloud_name.startswith('gs://')
        self.gcloud_name = gcloud_name
        bucket_name, blob_name = gcloud_name.split('gs://')[1].split('/', 1)
        bucket = storage.Client().get_bucket(bucket_name)
        self.blob = bucket.blob(blob_name)

    def __enter__(self):
        self.tempfile = tempfile.NamedTemporaryFile()
        return self.tempfile

    def __exit__(self, *args):
        self.tempfile.flush()
        print("UPLOADING TO {}".format(self.gcloud_name), flush=True)
        self.blob.upload_from_filename(self.tempfile.name)
        self.tempfile.close()


def ind_where(array: np.ndarray, target, return_first_match=True, default_value=-1):
    """
    :param array: Single dimension array
    :param target: target to search for
    :param return_first_match: If true, return the first index that matches, otherwise, return the last one
    :param default_value: Index to return if there was no match
    :return: index of the first match, or -1 if nothing
    """
    assert array.ndim == 1
    matching_inds = np.where(array == target)[0]
    if len(matching_inds) > 0:
        if return_first_match:
            return int(matching_inds[0])
        else:
            return int(matching_inds[-1])
    return default_value


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    news_config = GroverConfig.from_json_file(FLAGS.config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.iterations_per_loop,
        keep_checkpoint_max=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(news_config,
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=1e-4,
                                num_train_steps=0,
                                num_warmup_steps=0,
                                use_tpu=FLAGS.use_tpu,
                                )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size,
        params={'model_dir': FLAGS.output_dir}
    )

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        seq_length=FLAGS.max_seq_length,
        evaluate_for_fixed_number_of_steps=False,
        num_cpu_threads=1,
        is_training=False)
    result = [x for x in estimator.predict(input_fn=eval_input_fn, yield_single_examples=True)]
    cats = sorted(result[0].keys())
    result_stack = {cat: np.stack([x[cat] for x in result]) for cat in cats}

    with gcloudwriter(os.path.join(FLAGS.output_dir, FLAGS.validation_name)) as tempfile_name:
        with h5py.File(tempfile_name, 'w') as h5:
            for cat, data in result_stack.items():
                dtype2use = np.float16 if cat.endswith(('logprobs', 'top_p_required')) else np.uint16
                h5.create_dataset(cat, data=data.astype(dtype2use))
            h5.create_dataset('model', data=FLAGS.config_file)
            h5.create_dataset('ckpt', data=FLAGS.init_checkpoint)
            h5.create_dataset('input_file', data=FLAGS.input_file)

    # This gives the perplexity of the entire article. if you want to replicate the results of the paper you
    # might need to do something different to extract the ppl of just the body in particular.
    ppl_ex = []
    for logprobs_i, ids_i in zip(result_stack['gt_logprobs'], result_stack['labels']):
        # Omit the first token. Keep in mind input_ids is shifted by 1
        start_ind = ind_where(ids_i, target=50265, default_value=0)
        end_ind = ind_where(ids_i, target=50266, default_value=ids_i.shape[0] - 1)
        ppl_ex.append(logprobs_i[start_ind:end_ind])
    ppl_ex = np.concatenate(ppl_ex, 0)
    print("Article perplexity is {:.3f}".format(np.exp(-np.mean(ppl_ex))), flush=True)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
