# ImageNet_64x64 Generate 50000 samples
torchrun --standalone --nproc_per_node=4  generate.py   \
    --steps=20 --outdir=fid_tmp --seeds=100000-149999 --batch=256 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl

# FFHQ_64x64 Generate 50000 samples
torchrun --standalone --nproc_per_node=4  generate.py   \
    --steps=20 --outdir=fid_tmp --seeds=100000-149999 --batch=256 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl

# cifar-10 Generate 50000 samples
torchrun --standalone --nproc_per_node=4  generate.py  \
    --steps=20 --outdir=fid_tmp --seeds=100000-149999 --batch=1024 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl

# Test, Generate 32 samples
python generate.py --outdir=output --seeds=0-31 --steps=20 --batch=32 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl



