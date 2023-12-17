CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --mixed_precision="bf16" train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --center_crop \
  --train_data_dir="/home/jovyan/data/nhirschkind/icon_sample/train/" \
  --resolution=256 \
  --train_batch_size=24 \
  --num_train_epochs=20 --checkpointing_steps=200 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank 256
  --gradient_accumulation_steps 4 \
  --output_dir="/home/jovyan/data/nhirschkind/test_sd/$(date)/" \
  --validation_prompt='The thumbnail of a Roblox game called "Toilet Tower Obby"' \
  --report_to="tensorboard"