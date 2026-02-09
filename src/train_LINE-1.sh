arch=ribo-ld
rbz_type=LINE-1
python Ribo-LD-supervised.py --mode ae --arch $arch --data_path dataset/LINE-1.tsv  --max_epochs 100 --latent_dim 64 --seq_len 200
python Ribo-LD-supervised.py --mode diffusion_activity \
  --arch ${arch} \
  --data_path dataset/LINE-1.tsv \
  --latent_vectors_path vae_checkpoint/${arch}/latent_vectors_${arch}_${rbz_type}.npy \
  --ae_model_path vae_checkpoint/${arch}/best_model_${arch}_${rbz_type}.pt \
  --train_steps 20000 \
  --activity_eval_every 200 \
  --enable_dataset_optimization \
  --generation_count 768 \
  --latent_dim 64 \
  --learning_rate 1e-4 \
  --diffusion_lr 8e-5 \
  --act_ts 0.80 \
  --seq_len 200