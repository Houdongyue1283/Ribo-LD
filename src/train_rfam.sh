arch=ribo-ld-unsup
python Ribo-LD-unsupervised.py --mode ae --arch $arch --max_epochs 200 --latent_dim 128
python Ribo-LD-unsupervised.py --mode diffusion   \
  --arch $arch     \
  --latent_vectors_path vae_checkpoint/${arch}/latent_vectors_${arch}.npy     \
  --ae_model_path vae_checkpoint/${arch}/best_model_${arch}.pt     \
  --train_steps 40000   \
  --activity_eval_every 200 \
  --enable_dataset_optimization \
  --replacement_count 500 \
  --generation_count 1000 \
  --latent_dim 128\
  --learning_rate 1e-4\
  --diffusion_lr 8e-6