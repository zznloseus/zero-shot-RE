# source /home/lsl/anaconda3/envs/zeroRE/bin/activate zeroRE


# for dataset in 'fewrel'
# do
#   for unseen in 5
#   do
#     for seed in 7
#     do
#         for k in 2
#         do
#         python -u main2.py \
#         --gpu_available 1 \
#         --unseen ${unseen} \
#         --k ${k} \
#         --dataset ${dataset} \
#         --seed ${seed} \
#         --train_batch_size 32 \
#         --evaluate_batch_size 640 \
#         --epochs 5 \
#         --lr 2e-5 \
#         --warm_up 100 \
#         --pretrained_model_name_or_path ../../bert-base-uncased \
#         --add_auto_match True
#         done
#     done
#   done
# done
# 'wikizsl''fewrel' 19 
for dataset in 'fewrel'
do
  for unseen in 15
  do
    for seed in 7
    do
        for k in 2 
        do
        python -u main.py \
        --gpu_available 0 \
        --unseen ${unseen} \
        --k ${k} \
        --dataset ${dataset} \
        --seed ${seed} \
        --train_batch_size 32 \
        --evaluate_batch_size 640 \
        --epochs 5 \
        --lr 2e-5 \
        --warm_up 100 \
        --pretrained_model_name_or_path ../../bert-base-uncased \
        --add_auto_match True
        done
    done
  done
done