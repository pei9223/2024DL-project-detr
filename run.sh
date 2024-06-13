CUDA_VISIBLE_DEVICES=9 python main.py \
                            --batch_size 1 --no_aux_loss --eval \
                            --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
                            --coco_path /disk1/hyhsieh/coco \
                            --encoder_r 100 \
                            --decoder_r 0 \
                            --memory_r 0 \
                            --tome_vis

# python not_tracked_dir/line_notify.py