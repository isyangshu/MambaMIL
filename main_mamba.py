import os
import time
import sys

from datasets.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import CV_Meter

from torch.utils.data import DataLoader, SubsetRandomSampler

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        if args.model == "MambaMIL":
            results_dir = "./results/{dataset}/[{model}]-[{type}]-[{time}]-[{lr}]-[{layer}]-[{rate}]".format(
                type=args.mamba_type,
                dataset=args.excel_file.split('/')[-1].split('.')[0],
                model=args.model,
                time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
                lr=args.lr,
                layer=args.mamba_layer,
                rate=args.mamba_rate,
            )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    dataset = TCGA_Survival(excel_file=args.excel_file)
    args.num_classes = 4
    args.n_features = 1024
    # 5-fold cross validation
    meter = CV_Meter(fold=5)
    # start 5-fold CV evaluation.
    for fold in range(5):
        # get split
        train_split, val_split = dataset.get_split(fold)
        train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
        val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split))

        # build model, criterion, optimizer, schedular
        #################################################
        if args.model == "MambaMIL":
            from models.MambaMIL.network import MambaMIL
            from models.MambaMIL.engine import Engine
            model = MambaMIL(n_classes=args.num_classes, dropout=0.25, act="gelu", n_features=args.n_features, layer=args.mamba_layer, rate=args.mamba_rate, type=args.mamba_type)
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print('[model] trained model: ', args.model)
        criterion = define_loss(args)
        print('[model] loss function: ', args.loss)
        optimizer = define_optimizer(args, model)
        print('[model] optimizer: ', args.optimizer)
        scheduler = define_scheduler(args, optimizer)
        print('[model] scheduler: ', args.scheduler)
        # start training
        score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
        meter.updata(score, epoch)

    csv_path = os.path.join(results_dir, "results_{}.csv".format(args.model))
    meter.save(csv_path)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
