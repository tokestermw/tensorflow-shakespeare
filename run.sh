python -m tensorshake.translate.translate \
       --num_layers 2 \
       --size 256 \
       --learning_rate_decay_factor 0.5 \
       --train_dir /tmp/tensorflow
       # --decode 1 # turn on for prediction
