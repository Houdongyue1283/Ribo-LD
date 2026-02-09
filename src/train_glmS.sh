arch=ribo-ld
rbz_type=glmS
python Ribo-LD-supervised.py --mode ae --arch $arch --data_path dataset/glmS.tsv --max_epochs 100 --latent_dim 64 --seq_len 200
python Ribo-LD-supervised.py --mode diffusion_activity   \
  --arch ${arch}     \
  --data_path dataset/glms_log.tsv \
  --latent_vectors_path vae_checkpoint/${arch}/latent_vectors_${arch}_${rbz_type}.npy \
  --ae_model_path vae_checkpoint/${arch}/best_model_${arch}_${rbz_type}.pt \
  --train_steps 20000   \
  --activity_eval_every 200 \
  --enable_dataset_optimization \
  --generation_count 736 \
  --latent_dim 64\
  --learning_rate 1e-4\
  --diffusion_lr 8e-5\
  --lower_is_better\
  --act_ts 2.9 \
  --seq_len 200