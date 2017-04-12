pip install -r requirements.txt

# google seq2seq lib
git clone https://github.com/google/seq2seq.git
(cd seq2seq; pip install -e .)

# -- test
# https://google.github.io/seq2seq/getting_started/#common-installation-issues
python -m unittest seq2seq.test.pipeline_test
