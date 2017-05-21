#!/usr/bin/env bash
virtualenv .
#python -m venv .

source bin/activate

pip install -r requirements.txt
#pip install --upgrade \
# https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0rc1-py2-none-any.whl

# brew install bazel

# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo apt-get update && sudo apt-get install bazel

# -- deepmind sonnet lib
#git clone --recursive https://github.com/deepmind/sonnet
#pip install git+git://github.com/deepmind/sonnet.git@9d2098305cf49423e67529d0b5469e78f799843c

#(cd sonnet/tensorflow; git checkout r1.1; ./configure)
(cd sonnet/tensorflow; ./configure)

export SONNET_INSTALL_DIR="/tmp/sonnet"
(cd sonnet; mkdir -p ${SONNET_INSTALL_DIR})
(cd sonnet; bazel clean)
(cd sonnet; bazel build --config=opt :install)
(cd sonnet; ./bazel-bin/install ${SONNET_INSTALL_DIR})
(cd sonnet; pip install ${SONNET_INSTALL_DIR}/*.whl)

# -- test
python -c "
import sonnet as snt;
import tensorflow as tf;
out = snt.resampler(tf.constant([0.]), tf.constant([0.]));
print(out)"

 # -- google seq2seq lib
 # git clone https://github.com/google/seq2seq.git
 # (cd seq2seq; pip install -e .)

# # -- test
 # https://google.github.io/seq2seq/getting_started/#common-installation-issues
 # python -m unittest seq2seq.test.pipeline_test

# -- TensorFlow Serving
git clone -b 0.5.1 --recurse-submodules https://github.com/tensorflow/serving
