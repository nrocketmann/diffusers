export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="/home/jovyan/data/nhirschkind/icon_sample/train/"
export OUTPUT_DIR="/home/jovyan/data/nhirschkind/test_sd/$(date)/"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="title image of a sks game" \
  --class_prompt="title image of a game" \
  --resolution=256 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
