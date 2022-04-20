import numpy as np
from torch.utils.data import Dataset
from imageio import imread
from random import sample
from torch import tensor
import albumentations

import os


class HandGestures(Dataset):
    """
    Class for a custom data set.
    The class inherits from the abstract class torch.utils.data.Dataset
    It should at least implements implements __getitem__ and __len__
    """
    def __init__(self, path: os.path, files: list = None, transform: albumentations.Compose = None):
        self.path = path

        # If a list of files is not
        if files is None:
            self.files = [x for x in os.listdir(path) if x.endswith('jpg')]
        else:
            self.files = []
            for i in files:
                if i in os.listdir(path):
                    self.files.append(i)
                else:
                    raise FileExistsError(f'Image "{i}" does not exist in is directory {self.path}')
        classes = sorted(set([x[0] for x in self.files]))
        self.annotations = dict(zip(classes, range(len(classes))))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if type(item) is not slice:
            file = os.path.join(self.path, self.files[item])
            image = np.array(imread(file))
            sample_ = self.transform(image=image)['image'] if self.transform else image
            return sample_, tensor(self.annotations[self.files[item][0]])

        def func(x): return np.array(imread(os.path.join(self.path, x)))

        files = [(self.transform(image=func(x))['image'], tensor(self.annotations[x[0]]))
                 if self.transform else (func(x), tensor(self.annotations[x[0]]))
                 for x in self.files[item]]
        return files

    def train_test_split(self, test_size=.2, stratify=True, test_transform=None):
        if not stratify:
            len_test = round(test_size * len(self.files))
            test_files = sample(self.files, k=len_test)
        else:
            test_files = []
            for i in self.annotations:
                category = [x for x in self.files if x.startswith(i)]
                len_test = round(test_size * len(category))
                test_category = sample(category, k=len_test)
                test_files += test_category
        train_files = [x for x in self.files if x not in test_files]
        train_set = HandGestures(self.path, files=train_files, transform=self.transform)
        test_transform = self.transform if test_transform is None else test_transform
        test_set = HandGestures(self.path, files=test_files, transform=test_transform)
        return train_set, test_set


if __name__ == '__main__':
    ...
