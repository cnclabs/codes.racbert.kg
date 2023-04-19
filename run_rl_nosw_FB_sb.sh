#!/bin/sh

# _degree_ada linearly adapting the avg weights
# avg
# our

# SRPRS/en_fr data/fb_dbp
CUDA_VISIBLE_DEVICES=0 python3 ls_mab_comb_ori_Comb-decodeconcat-tunelast19-round5-direct.py --log gcnalign \
                                    --model_name "GCNAPP_contras_active-decodeconcat-SB-Iter-epoch3-tunelast19-round5-direct" \
                                    --finetune_dataset "FB100-SB-Iter-epoch3-tunelast19-round5" \
                                    --finetune_epoch 3 \
                                    --sb_fine_tune 1 \
                                    --load_from_ori 0 \
                                    --seed 2020\
                                    --data_dir "data/fb_dbp" \
                                    --rate 0.3 \
                                    --epoch 300 \
                                    --check 300 \
                                    --update 10 \
                                    --train_batch_size -1 \
                                    --encoder "GCN-Align" \
                                    --encoder1 "APP" \
                                    --hiddens "100,100,100" \
                                    --heads "2,2" \
                                    --decoder "Align" \
                                    --sampling "N" \
                                    --k "25" \
                                    --margin "1" \
                                    --alpha "1" \
                                    --feat_drop 0.0 \
                                    --lr 0.002\
                                    --train_dist "euclidean" \
                                    --test_dist "euclidean" \
                                    --mytest True \
                                    --sbert True \
                                    --sb_w "w1" \
                                    --neg_scale 1