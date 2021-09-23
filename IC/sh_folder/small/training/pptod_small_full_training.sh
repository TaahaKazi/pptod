CUDA_VISIBLE_DEVICES=6 python ../../../learn.py\
    --data_prefix ../../../../data/banking77/\
    --datapoints_per_intent 10000\
    --num_train_epochs 100\
    --model_name t5-small\
    --format_mode ic\
    --pretrained_path ../../../../checkpoints/small/\
    --batch_size_per_gpu 128\
    --number_of_gpu 1\
    --gradient_accumulation_steps 1\
    --save_path ../../../ckpt/small/full_training/