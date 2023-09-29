# ImageNet_256x256, class-conditional
mpiexec -n 4 python er_sde_sample.py --attention_resolutions 32,16,8 --class_cond True  --diffusion_steps 20 \
        --batch_size 32 --num_samples 50000  --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --model_path models/256x256_diffusion.pt

# ImageNet_256x256, classifier guidance
mpiexec -n 4 python classifier_er_sde_sample.py --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 20 \
        --batch_size 20 --num_samples 50000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --classifier_scale 2.0 --classifier_path models/256x256_classifier.pt \
        --model_path models/256x256_diffusion.pt

# ImageNet_128x128, class-conditional
mpiexec -n 4 python er_sde_sample.py --attention_resolutions 32,16,8 --class_cond True  --diffusion_steps 20 \
        --batch_size 128 --num_samples 50000  --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --model_path models/128x128_diffusion.pt

# ImageNet_128x128, classifier guidance
mpiexec -n 4 python classifier_er_sde_sample.py --attention_resolutions 32,16,8 --class_cond True  --diffusion_steps 20 \
        --batch_size 64 --num_samples 50000  --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --classifier_scale 2.0 --classifier_path models/128x128_classifier.pt \
        --model_path models/128x128_diffusion.pt

# LSUN bedroom 256x256, unconditional
mpiexec -n 4 python er_sde_sample.py --attention_resolutions 32,16,8 --class_cond False  --diffusion_steps 20 \
        --batch_size 32 --num_samples 50000  --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --model_path models/lsun_bedroom.pt

# Test, Generate 16 samples
python er_sde_sample.py --attention_resolutions 32,16,8 --class_cond True  --diffusion_steps 20 \
       --batch_size 4 --num_samples 16  --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True\
       --model_path models/128x128_diffusion.pt





 