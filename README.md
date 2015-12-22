# Shakespeare translations using TensorFlow

This is an example of using the new Google's [TensorFlow](https://github.com/tensorflow/tensorflow) library on
monolingual translation going from modern English to Shakespeare based on research from 
[Wei Xu](https://github.com/cocoxu/Shakespeare).

## Prepare

First download the TensorFlow library depending on your platform:

```
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl # for mac
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl # for ubuntu
```

1. Grabs parallel data.
2. Gets train, dev split.
3. Builds vocabulary
4. Converts parallel data into ids

From the root directory:

```
python -m tensorshake.get_data
python -m tensorshake.prepare_corpus
```

Delete /cache to start anew.

## Train

Use the example BASH script to train the model. This saves the check points in the `--train_dir` directory.
If you run it again, the training process continues from the check point. To restart with fresh parameters,
simply delete/rename the check points.

```
./run.sh
```

## Results

[Benchmarks](http://aclweb.org/anthology/C/C12/C12-1177.pdf) (Shakespeare -> Modern English)

Input | Output
 --- | ---
i will bite thee by the ear for that jest . | i ’ ll bite you by the ear for that joke .
what further woe conspires against mine age ? | what ’ s true despair conspires against my old age ?
how doth my lady ? |how is my lady ?
hast thou slain tybalt ? |have you killed tybalt ?
an i might live to see thee married once , i have my wish .| if i could live to see you married, i ’ ve my wish .
benvolio , who began this bloody fray ? | benvolio , who started this bloody fight itself ?
what is your will ? | what do you want ? 
call her forth to me . |bring her out to me .

*This repo* (Modern English -> Shakespeare)

| Input | Output 
----- | ---

## Pre-Trained Models

Here is a pre-trained model. Download both files into a directory and point to it using `--train_dir` and 
predict using `--decode` option.

https://s3-us-west-2.amazonaws.com/tensorflow-shakespeare/example/checkpoint

https://s3-us-west-2.amazonaws.com/tensorflow-shakespeare/example/translate.ckpt-10400


## Possible improvements

- word embeddings
- beam search
- language model reranking