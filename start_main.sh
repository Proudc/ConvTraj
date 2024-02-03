dataset_list=(0_geolife 0_porto_all)
dist_type_list=(frechet haus dtw edr)
seed_list=(666 555 444)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for dist_type in ${dist_type_list[@]}; do
            train_flag=${dataset}_${dist_type}_${seed}
            echo ${train_flag}
            nohup python train.py --train_flag ${train_flag} --dist_type ${dist_type} --random_seed ${seed} --test_epoch 200 --epoch_num 200 --device cuda:0 --root_read_path /mnt/data_hdd1/czh/Neutraj/${dataset} --root_write_path /mnt/data_hdd1/czh/Neutraj/${dataset} > train_log/${train_flag} &
            PID0=$!
            wait $PID0
        done
    done
done