#!/bin/bash

source ~/.profile

python scripts/run_model.py \
  --program_generator "data/film2.pt" \
  --execution_engine "data/film2.pt" \
  --input_question_h5 "data/val_human_questions.h5" \
  --input_features_h5 "data/val_features.h5" \
  --vocab_json "data/human_vocab.json" \
  --output_preds "preds.txt"

#python scripts/run_model.py \
#  --checkpoint_path $checkpoint_path \
#  --num_train_samples 256 \
#  --model_type FiLM \
#  --num_iterations 1000 \
#  --print_verbose_every 5000 \
#  --checkpoint_every 278 \
#  --record_loss_every 278 \
#  --num_val_samples 149991 \
#  --batch_size 64 \
#  --use_coords 1 \
#  --module_stem_batchnorm 1 \
#  --module_stem_num_layers 1 \
#  --module_batchnorm 1 \
#  --classifier_batchnorm 1 \
#  --bidirectional 0 \
#  --decoder_type linear \
#  --encoder_type gru \
#  --rnn_num_layers 1 \
#  --rnn_wordvec_dim 200 \
#  --rnn_hidden_dim 4096 \
#  --rnn_output_batchnorm 0 \
#  --classifier_downsample maxpoolfull \
#  --classifier_proj_dim 512 \
#  --classifier_fc_dims 1024 \
#  --module_input_proj 1 \
#  --module_residual 1 \
#  --module_dim 128 \
#  --module_dropout 0e-2 \
#  --module_stem_kernel_size 3 \
#  --module_kernel_size 3 \
#  --module_batchnorm_affine 0 \
#  --module_num_layers 1 \
#  --num_modules 4 \
#  --condition_pattern 1,1,1,1 \
#  --gamma_option linear \
#  --gamma_baseline 1 \
#  --use_gamma 1 \
#  --use_beta 1 \
#  --program_generator "data/film_humans_256.pt" \
#  --execution_engine "data/film_humans_256.pt" \
#  --optimizer Adam \
#  --learning_rate 3e-4 \
#  --weight_decay 1e-5 \
#  --train_program_generator 1 \
#  --train_execution_engine 0 \
#  --set_execution_engine_eval 0 \
#  --train_question_h5 "data/train_human_questions.h5" \
#  --val_question_h5 "data/val_human_questions.h5" \
#  --vocab_json "data/human_vocab.json" \
#  | tee $log_path
