model_names='max_mil mean_mil att_mil trans_mil s4_mil mamba_mil'

backbones="resnet50 plip"



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

task="LUAD_LUSC"
# results_dir="/jhcnas3/Pathology/experiments/train_vl/"$task
results_dir="./experiments/train/"$task
model_size="small" # since the dim of feature of vit-base is 768    
preloading="no"
patch_size="512"


lr='2e-4'
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'

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
        python main.py \
            --drop_out 0\
            --early_stopping \
            --lr $lr \
            --k 10 \
            --reg 1e-4 \
            --k_start $k_start \
            --k_end $k_end \
            --label_frac 1.0 \
            --exp_code $exp \
            --patch_size $patch_size \
            --weighted_sample \
            --bag_loss ce \
            --inst_loss svm \
            --task $task \
            --backbone $backbone \
            --results_dir $results_dir \
            --model_type $model \
            --log_data \
            --split_dir './splits/LUAD_LUSC_100' \
            --preloading $preloading \
            --model_size $model_size \
            --in_dim ${in_dim[$backbone]} \
            --mambamil_rate $mambamil_rate \
            --mambamil_layer $mambamil_layer \
            --mambamil_type $mambamil_type
    done
done

