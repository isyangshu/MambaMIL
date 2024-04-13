# studies="BLCA BRCA CESC CRC GBMLGG KIRC LIHC LUAD LUSC PAAD SARC UCEC"
studies="BLCA"
models="MambaMIL"

for study in $studies
do
    for model in $models
    do
        CUDA_VISIBLE_DEVICES=0 python main_mamba.py --model $model \
        --lr 2e-5 \
        --mamba_type SRMamba \
        --mamba_layer 2 \
        --mamba_rate 10 \
        --excel_file /home/syangcw/MambaMIL/csv/Cbioportal/${study}_Splits.csv \
        --num_epoch 30 \
        --batch_size 1
    done
done
