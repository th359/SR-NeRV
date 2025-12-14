base_path=dataset
sr_path=../sr/output/best_model.pth
method=NeRV
# method=HNeRV

# datasets=(
#     bear           blackswan        bmx-bumps          bmx-trees       boat
#     breakdance     breakdance-flare bus                camel           car-roundabout
#     car-shadow     car-turn         cows               dance-jump      dance-twirl
#     dog            dog-agility      drift-chicane      drift-straight  drift-turn
#     elephant       flamingo         goat               hike            hockey
#     horsejump-high horsejump-low    kite-surf          kite-walk       libby
#     lucia          mallard-fly      mallard-water      motocross-bumps motocross-jump
#     motorbike      paragliding      paragliding-launch parkour         rhino
#     rollerblade    scooter-black    scooter-gray       soapbox         soccerball
#     stroller       surf             swing              tennis          train
# )
datasets=(bear)

for video in "${datasets[@]}"
do
python train_nerv_all.py  \
    --method ${method} --outf davis  --exp_id ${video} --data_path ${base_path}/DAVIS-data/DAVIS/JPEGImages/1080p/${video} --vid ${video}   \
    --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 2 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.38  -e 300 --eval_freq 1000  --lower_width 12 -b 1 --lr 0.001 \
    --scale 2 --sr_weight ${sr_path} --finetune_epoch 6 --dump_images
done
