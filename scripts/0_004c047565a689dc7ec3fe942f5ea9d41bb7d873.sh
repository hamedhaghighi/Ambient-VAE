HPARAMS="
dataset=mnist,\
measurement_type=blur_addnoise,\
drop_prob=0.0,\
patch_size=14,\
blur_radius=1.0,\
blur_filter_size=5,\
additive_noise_std=0.5,\
num_angles=1,\
\
train_mode=baseline,\
unmeasure_type=None,\
\
model_class=unconditional,\
model_type=wgangp,\
z_dim=128,\
z_dist=uniform,\
gp_lambda=10.0,\
d_ac_lambda=1.0,\
g_ac_lambda=0.1,\
\
opt_type=adam,\
batch_size=64,\
g_lr=0.0002,\
d_lr=0.0002,\
lr_decay=false,\
opt_param1=0.5,\
opt_param2=0.9999,\
g_iters=1,\
d_iters=5,\
\
results_dir=./results/,\
sample_num=100,\
max_checkpoints=1,\
max_train_iter=27000,\
"

python src/main.py \
    --hparams $HPARAMS