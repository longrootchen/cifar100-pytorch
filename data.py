import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transforms():
    normalize = transforms.Normalize(mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276))
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def get_valid_transforms():
    normalize = transforms.Normalize(mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276))
    return transforms.Compose([transforms.ToTensor(), normalize])


class CIFAR100Dataset(Dataset):

    def __init__(self, df, img_dir, phase='train'):
        assert phase in ('train', 'val')
        self.img_dir = img_dir
        self.df = df
        self.phase = phase

        if phase == 'train':
            self.transform = get_train_transforms()
        elif phase == 'val':
            self.transform = get_valid_transforms()

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df.iloc[index]['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.df.iloc[index]['fine_target']
        return image, label

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    import pandas as pd
    from matplotlib import pyplot as plt

    train_dir = os.path.join(os.curdir, 'datasets', 'train')
    df = pd.read_csv(os.path.join(os.curdir, 'datasets', 'train_folds.csv'))
    train_df = df[df['fine_fold'] != 1]
    valid_df = df[df['fine_fold'] == 1]
    train_set = CIFAR100Dataset(train_df, train_dir, 'train')
    valid_set = CIFAR100Dataset(valid_df, train_dir, 'val')

    test_dir = os.path.join(os.curdir, 'datasets', 'test')
    test_df = pd.read_csv(os.path.join(os.curdir, 'datasets', 'test.csv'))
    test_set = CIFAR100Dataset(test_df, test_dir, 'val')

    print(len(train_set), len(valid_set), len(test_set))
