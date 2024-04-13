import pandas as pd
import torch
import torch.utils.data as data

class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, signatures=None):
        self.signatures = signatures
        print('[dataset] loading dataset from %s' % (excel_file))
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(self.rows)))

    def get_split(self, fold=0):
        assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
        split = self.rows['Fold {}'.format(fold)].values.tolist()
        train_split = [i for i, x in enumerate(split) if x == 'train: complete']
        val_split = [i for i, x in enumerate(split) if x == 'val: complete']
        print("[dataset] (fold {}) training split (exclude missing data): {}, validation split: {}".format(fold, len(train_split), len(val_split)))
        return train_split, val_split

    def read_WSI(self, path):
        path = path.replace("/storage2/pathology/", "/project/mmendoscope/WSI/")
        wsi = [torch.load(x) for x in path.split(';')]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI, RNA = case[:6]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0

        WSI = self.read_WSI(WSI)
        return (ID, WSI, Event, Censorship, Label)

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows