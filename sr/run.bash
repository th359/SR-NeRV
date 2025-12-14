data_path=dataset/DIV2K
out_path=output

python train.py --data_dir ${data_path} \
                --scale 2 --patch_size 160 --batch_size 64 \
                --epochs 5000 --eval_interval 30 --device cuda \
                --output ${out_path} \
                --lr 8e-4 --warmup_epochs 5 --lr_milestones 60 150 240 330 420 --lr_gamma 0.5
