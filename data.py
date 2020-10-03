from torchvision import datasets, transforms


def cifar100_train_set(data_root='./datasets'):
    """get CIFAR-100 training dataset, following the common-used data augmentation"""
    print('Building CIFAR-100 Dataset.')
    dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ]))
    return dataset


def cifar100_test_set(data_root='./datasets'):
    """get CIFAR-100 testnig dataset"""
    print('Building CIFAR-100 Dataset.')
    dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ]))
    return dataset


if __name__ == '__main__':
    train_set = cifar100_train_set()
    test_set = cifar100_test_set()

    print(len(train_set), len(test_set))
