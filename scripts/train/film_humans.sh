#!/bin/bash

source ~/.profile

TRAIN_SIZE=4096
let "TRAIN_ITERS = $TRAIN_SIZE * 4"
echo "$TRAIN_SIZE $TRAIN_ITERS"

unk_threshold=5
checkpoint_path="data/film_humans_$TRAIN_SIZE.pt"
log_path="data/film_humans_$TRAIN_SIZE.log"
python scripts/train_model.py \
  --checkpoint_path $checkpoint_path \
  --num_train_samples $TRAIN_SIZE \
  --model_type FiLM \
  --num_iterations $TRAIN_ITERS \
  --print_verbose_every 5000 \
  --checkpoint_every 278 \
  --record_loss_every 278 \
  --num_val_samples 149991 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 4096 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 128 \
  --module_dropout 0e-2 \
  --module_stem_kernel_size 3 \
  --module_kernel_size 3 \
  --module_batchnorm_affine 0 \
  --module_num_layers 1 \
  --num_modules 4 \
  --condition_pattern 1,1,1,1 \
  --gamma_option linear \
  --gamma_baseline 1 \
  --use_gamma 1 \
  --use_beta 1 \
  --program_generator_start_from data/film2.pt \
  --execution_engine_start_from data/film2.pt \
  --optimizer Adam \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --train_program_generator 1 \
  --train_execution_engine 0 \
  --set_execution_engine_eval 0 \
  --train_question_h5 "data/train_human_questions.h5" \
  --val_question_h5 "data/val_human_questions.h5" \
  --vocab_json "data/human_vocab.json" \
  | tee $log_path
