model_names='max_mil mean_mil att_mil trans_mil s4_mil mamba_mil'
# model_names='mamba_mil'
backbones="resnet50 plip"
# backbones='resnet50'

declare -A in_dim
in_dim["resnet50"]=1024
in_dim["plip"]=512

declare -A gpus
gpus["mean_mil"]=0
gpus["max_mil"]=0
gpus["att_mil"]=0
gpus["trans_mil"]=0
gpus['s4model']=0
gpus['mamba_mil']=0

cancers='BLCA BRCA COADREAD KIRC KIRP LUAD STAD UCEC'

lr='2e-4'
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'

for cancer in $cancers
    do
    task="TCGA_${cancer}_survival"
    data_root_dir="/data/wangyihui/feature/${cancer}"
    results_dir="./experiments/train/"$task
    preloading="no"
    patch_size="512"
    for model in $model_names
    do
        for backbone in $backbones
        do
            exp=$model"/"$backbone
            echo $exp", GPU is:"${gpus[$model]}
            export CUDA_VISIBLE_DEVICES=${gpus[$model]}
            # k_start and k_end, only for resuming, default is -1
            k_start=-1
            k_end=-1
            python main_survival.py \
                --drop_out 0.25\
                --early_stopping \
                --lr $lr \
                --k 5 \
                --k_start $k_start \
                --k_end $k_end \
                --label_frac 1.0 \
                --exp_code $exp \
                --patch_size $patch_size \
                --batch_size 1 \
                --weighted_sample \
                --bag_loss nll_surv \
                --task $task \
                --backbone $backbone \
                --results_dir $results_dir \
                --model_type $model \
                --log_data \
                --split_dir "./splits/TCGA_${cancer}_survival_kfold" \
                --data_root_dir $data_root_dir \
                --preloading $preloading \
                --in_dim ${in_dim[$backbone]} \
                --k_fold True \
                --mambamil_rate $mambamil_rate \
                --mambamil_layer $mambamil_layer \
                --mambamil_type $mambamil_type
        done
    done
done
