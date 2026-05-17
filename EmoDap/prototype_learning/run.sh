for dataset in 'ED'
do
  for unseen in 21
  do
    for seed in 7
    do
      for topk in 4
      do
        for run_seed in 1
        do
        python -u main_candi.py \
        --gpu_available 0 \
        --unseen ${unseen} \
        --dataset ${dataset} \
        --seed ${seed} \
        --topk ${topk} \
        --run_seed ${run_seed} \
        --train_batch_size 16 \
        --evaluate_batch_size 1 \
        --epochs 20 \
        --lr 1e-5 \
        --warm_up 100 \
        --pretrained_model_name_or_path /your_path/all-mpnet-base-v2 \
        --add_auto_match False
        done
      done
    done
  done
done
#16, 40
#'dialogues', 'emory', 'iemocap', 'meld'
