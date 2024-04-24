from __future__ import print_function

import argparse
import os
from timeit import default_timer as timer

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.survival_core_utils import train
from dataset.dataset_survival import Generic_MIL_Survival_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np
import wandb

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    wandb.init(project=args.task)
    wandb.config.update(args)
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_test_cindex = []
    latest_val_cindex = []
    
    folds = np.arange(start, end)
    
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        if args.k_fold:
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        else: 
            print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        if args.k_fold:
            datasets = (train_dataset, val_dataset)
        else:
            datasets = (train_dataset, val_dataset, test_dataset)
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()
                
        if args.task_type == 'survival':
            if args.k_fold:
                cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
            else:
                results, cindex_test, cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
                latest_test_cindex.append(cindex_test)
            
        # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        # all_test_auc.append(test_auc)
        # all_val_auc.append(val_auc)
        # all_test_acc.append(test_acc)
        # all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not args.k_fold:
            save_pkl(filename, results)
    if args.k_fold:
        final_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    else: 
        final_df = pd.DataFrame({'folds': folds, 'test_cindex': latest_test_cindex, 
            'val_cindex': latest_val_cindex, })
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    if not args.k_fold:
        mean_test = final_df['test_cindex'].mean()
        std_test = final_df['test_cindex'].std()
    mean_val = final_df['val_cindex'].mean()
    std_val = final_df['val_cindex'].std()

    if args.k_fold:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'val_cindex': [mean_val, std_val]
        })
    else:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'test_cindex': [mean_test, std_test],
            'val_cindex': [mean_val, std_val]
        })
    final_df = final_df.append(df_append, ignore_index=True)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary": table})
    if args.k_fold:
        wandb.log({"mean_val_cindex": mean_val})
    else:
        wandb.log({"mean_test_cindex": mean_test, "mean_val_cindex": mean_val})

    
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='Data directory to WSI features (extracted via CLAM)')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=1,)
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['mean_mil', 'max_mil', 'att_mil','trans_mil', 's4model','mamba_mil'], default='mamba_mil', 
                    help='type of model')
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: None).')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='')
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--k_fold', type=bool, default=False, help='k fold for cross validation')


## mambamil

parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Deviece is:', device)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}


print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    study = '_'.join(args.task.split('_')[:2])
    
    combined_study = study
    combined_study = combined_study.split('_')[1]
    # study_dir = '%s_20x_features' % combined_study
    study_dir = 'pt_files/%s' % args.backbone
    dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/%s_processed.csv' % combined_study,
                                            mode = args.mode,
                                            apply_sig = args.apply_sig,
                                            data_dir= os.path.join(args.data_root_dir, study_dir), #! cluster.pkl should be as same as data_dir
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
else:
	raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError
    
# if not os.path.exists(args.results_dir):
#     os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))

