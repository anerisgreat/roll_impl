import scipy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import os
import keel_ds

#for Adult
from adult import Adult

import torch
import torchvision
import torchvision.transforms as transforms

class ForestCoverDataset:
    def __init__(self):
        PATH = '/home/aner/.data/forestcover/cover.mat'

        mat = scipy.io.loadmat(PATH)

        x = mat['X']
        y = mat['y']

        self.x = \
            torch.tensor(PowerTransformer().fit_transform(x),
                         dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

class Cifar10Dataset:
    def __init__(self):
        PATH = '/home/aner/.data/cifar10'
        self._transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._dset = torchvision.datasets.CIFAR10(
            root=PATH, train=True, download=True, transform=self._transform)
        self.y = torch.tensor(torch.tensor(self._dset.targets) == 1, dtype = torch.float64)

    def __getitem__(self, i):
        x = self._dset.data[i]
        x = torch.tensor(x / 255, dtype = torch.float32)
        x = torch.movedim(x, -1, -3)
        return x, self.y[i]


        # if(type(i) == int):
        #     return self._dset.data[i], self.y[i]
        # if(len(i.shape) > 0):
        #     x, __= list(zip(*[self._dset.data[j] for j in i]))
        #     return torch.stack(x, 0), self.y[i]
        # else:
        #     return self._dset.data[i], self.y[i]

    def __len__(self):
        return len(self._dset)

class TestGaussianDataset:
    def __init__(self,
            loc_false, scale_false, n_false,
            loc_true, scale_true, n_true):
        false_s = np.random.normal(loc_false, scale_false,
                                   np.array([n_false, len(loc_false)]))
        true_s = np.random.normal(loc_true, scale_true,
                                  np.array([n_true, len(loc_true)]))

        samps = np.concatenate((false_s, true_s))

        labels = np.concatenate((
            np.zeros(n_false, dtype = bool),
            np.ones(n_true, dtype = bool)))

        self.x = torch.tensor(samps, dtype = torch.float32)
        self.y = torch.tensor(labels, dtype = torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

class AdultDataset:
    def __init__(self):
        self._data = Adult(root = 'datasets', download = True)
        self.x, self.y = self._data[:]
        self.y = self.y.float()

    def __getitem__(self, i):
        retx, rety = self._data[i]
        return retx, rety.float()

    def __len__(self, i):
        return len(self._data)

KEEL_TYPE_MAP = {
    'integer' : int,
    'real' : float,
    'Class' : str
    }

class TorchStandardScaler:
    def fit(self, x):
        """
        Calculates the mean and standard deviation of the input tensor x.
        """
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        """
        Applies the standardization transform to the input tensor x.
        """
        x -= self.mean
        x /= (self.std + 1e-7)  # Add a small epsilon to prevent division by zero
        return x

    def fit_transform(self, x):
        """
        Fits the scaler and then transforms the input tensor.
        """
        self.fit(x)
        return self.transform(x)

class KeelDataset:
    def __init__(self, dset_name, type_data='imbalanced'):
        keel_data = keel_ds.load_data(dset_name, type_data=type_data)
        fold = keel_data[0]
        self.x = TorchStandardScaler().fit_transform(
            torch.from_numpy(np.concatenate((fold[0], fold[2])).astype(float)).float())
        self.y = torch.from_numpy(np.concatenate((fold[1], fold[3])).astype(float)).float()

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

def parse_keel_dat(file_path):
    attributes = []
    data = []
    in_data_section = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.lower().startswith('@relation'):
                # Process relation name if needed
                pass
            elif line.lower().startswith('@attribute'):
                # Parse attribute definition
                parts = line.split()
                attr_name = parts[1]
                attr_type = parts[2]
                attributes.append({'name': attr_name, 'type': attr_type})
            elif line.lower().startswith('@data'):
                in_data_section = True
            elif in_data_section and line:
                # Parse data rows (assuming comma-separated values)
                data.append([item.strip() for item in line.split(',')])
    attribute_name_type_map = dict(map(lambda d: tuple(d['name'], KEEL_TYPE_MAP[d['type']]), attributes))
    return attributes, data

def get_keel_dataset():
    file_path = os.getenv('keel_wisconsin_dir')
    attributes, data = parse_keel_dat(file_path)

    print("Attributes:", attributes)
    print("Data (first 5 rows):", data[:5])


class BankMarketingDataset:
    def __init__(self):
        data_dir = os.getenv('uci_bank_additional_dir')
        if data_dir is None:
            raise ValueError("UCI Bank Marketing dataset not found. Set uci_bank_additional_dir environment variable.")

        import pandas as pd
        csv_path = os.path.join(data_dir, 'bank-additional-full.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_dir, 'bank-full.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_dir, 'bank.csv')

        df = pd.read_csv(csv_path, sep=';')
        df['y'] = (df['y'] == 'yes').astype(int)

        categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                           'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        feature_cols = [c for c in df.columns if c != 'y']
        X = df[feature_cols].values.astype(np.float32)
        y = df['y'].values.astype(np.float32)

        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self._scaler = TorchStandardScaler()
        self.x = self._scaler.fit_transform(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class HiggsDataset:
    def __init__(self, n_samples=500_000):
        import pandas as pd
        data_dir = os.getenv('uci_higgs_dir')
        if data_dir is None:
            raise ValueError("HIGGS dataset not found. Set uci_higgs_dir environment variable.")

        df = pd.read_csv(os.path.join(data_dir, 'HIGGS.csv.gz'),
                         header=None, compression='gzip')

        if n_samples is not None and n_samples < len(df):
            pos = df[df[0] == 1.0].sample(n=n_samples // 2, random_state=42)
            neg = df[df[0] == 0.0].sample(n=n_samples // 2, random_state=42)
            df = pd.concat([pos, neg]).sample(frac=1, random_state=42)

        y = df.iloc[:, 0].values.astype(np.float32)
        X = df.iloc[:, 1:].values.astype(np.float32)

        self.x = TorchStandardScaler().fit_transform(
            torch.tensor(X, dtype=torch.float32))
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class CreditCardFraudDataset:
    def __init__(self):
        import pandas as pd
        data_dir = os.getenv('credit_card_fraud_dir')
        if data_dir is None:
            raise ValueError(
                "Credit card fraud dataset not found.\n"
                "  Set credit_card_fraud_dir environment variable, or use `nix develop` which sets it automatically.\n"
                "  Expected file: $credit_card_fraud_dir/creditcard.csv\n"
                "  Default path:  ~/.data/creditcard/creditcard.csv\n"
                "  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "  (Kaggle account required — download creditcard.csv and place it at the path above)"
            )

        df = pd.read_csv(os.path.join(data_dir, 'creditcard.csv'))

        y = df['Class'].values.astype(np.float32)
        feature_cols = [c for c in df.columns if c != 'Class']
        X = df[feature_cols].values.astype(np.float32)

        x = torch.tensor(X, dtype=torch.float32)
        # V1-V28 are already PCA-scaled; only scale Time and Amount (cols 0 and 29)
        scaler = TorchStandardScaler()
        x[:, [0, 29]] = scaler.fit_transform(x[:, [0, 29]])

        self.x = x
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class HomeCreditDataset:
    def __init__(self):
        import pandas as pd
        data_dir = os.getenv('home_credit_dir')
        if data_dir is None:
            raise ValueError(
                "Home Credit dataset not found.\n"
                "  Set home_credit_dir environment variable, or use `nix develop` which sets it automatically.\n"
                "  Expected file: $home_credit_dir/application_train.csv\n"
                "  Default path:  ~/.data/homecredit/application_train.csv\n"
                "  Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data\n"
                "  (Kaggle account required — download application_train.csv and place it at the path above)"
            )

        df = pd.read_csv(os.path.join(data_dir, 'application_train.csv'))

        y = df['TARGET'].values.astype(np.float32)
        df = df.drop(columns=['TARGET', 'SK_ID_CURR'])
        df = df.select_dtypes(include=[np.number])
        df = df.fillna(0)

        X = df.values.astype(np.float32)
        self.x = TorchStandardScaler().fit_transform(
            torch.tensor(X, dtype=torch.float32))
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
