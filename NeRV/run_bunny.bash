base_path=dataset
sr_path=../sr/output/best_model.pth
method=NeRV
# method=HNeRV

python train_nerv_all.py  \
    --method ${method} --outf bunny  --exp_id bunny --data_path ${base_path}/bunny --vid bunny   \
    --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 2 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.38  -e 300 --eval_freq 300  --lower_width 12 -b 1 --lr 0.001 \
    --scale 2 --sr_weight ${sr_path} --finetune_epoch 6 --dump_images
