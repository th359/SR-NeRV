base_path=dataset
sr_path=../sr/output/best_model.pth
method=NeRV
# method=HNeRV

datasets=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

for data in "${datasets[@]}"
do
python train_nerv_all.py  \
    --method ${method} --outf mcl/${method}  --exp_id mcl_${data}_${method}  \
    --data_path ${base_path}/MCL-JCV/images/1080PAVCFQPvideoSRC${data} --vid mcl_${data}_${method}   \
    --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 2 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.38  -e 300 --eval_freq 300  --lower_width 12 -b 1 --lr 0.001 \
    --scale 2 --sr_weight ${sr_path} --finetune_epoch 6 --dump_images
done
