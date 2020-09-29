from torchvision import datasets, transforms

coarse_cls_to_idx = {
    'aquatic_mammals': 0, 'fish': 1, 'flowers': 2, 'food_containers': 3, 'fruit_and_vegetables': 4,
    'household_electrical_devices': 5, 'household_furniture': 6, 'insects': 7,
    'large_carnivores': 8, 'large_man_made_outdoor_things': 9,
    'large_natural_outdoor_scenes': 10, 'large_omnivores_and_herbivores': 11,
    'medium_mammals': 12, 'non_insect_invertebrates': 13, 'people': 14, 'reptiles': 15,
    'small_mammals': 16, 'trees': 17, 'vehicles_1': 18, 'vehicles_2': 19
}

fine_cls_to_idx = {
    'beaver': 0, 'dolphin': 1, 'otter': 2, 'seal': 3, 'whale': 4,
    'aquarium_fish': 5, 'flatfish': 6, 'ray': 7, 'shark': 8, 'trout': 9,
    'orchid': 10, 'poppy': 11, 'rose': 12, 'sunflower': 13, 'tulip': 14,
    'bottle': 15, 'bowl': 16, 'can': 17, 'cup': 18, 'plate': 19,
    'apple': 20, 'mushroom': 21, 'orange': 22, 'pear': 23, 'sweet_pepper': 24,
    'clock': 25, 'keyboard': 26, 'lamp': 27, 'telephone': 28, 'television': 29,
    'bed': 30, 'chair': 31, 'couch': 32, 'table': 33, 'wardrobe': 34,
    'bee': 35, 'beetle': 36, 'butterfly': 37, 'caterpillar': 38, 'cockroach': 39,
    'bear': 40, 'leopard': 41, 'lion': 42, 'tiger': 43, 'wolf': 44,
    'bridge': 45, 'castle': 46, 'house': 47, 'road': 48, 'skyscraper': 49,
    'cloud': 50, 'forest': 51, 'mountain': 52, 'plain': 53, 'sea': 54,
    'camel': 55, 'cattle': 56, 'chimpanzee': 57, 'elephant': 58, 'kangaroo': 59,
    'fox': 60, 'porcupine': 61, 'possum': 62, 'raccoon': 63, 'skunk': 64,
    'crab': 65, 'lobster': 66, 'snail': 67, 'spider': 68, 'worm': 69,
    'baby': 70, 'boy': 71, 'girl': 72, 'man': 73, 'woman': 74,
    'crocodile': 75, 'dinosaur': 76, 'lizard': 77, 'snake': 78, 'turtle': 79,
    'hamster': 80, 'mouse': 81, 'rabbit': 82, 'shrew': 83, 'squirrel': 84,
    'maple_tree': 85, 'oak_tree': 86, 'palm_tree': 87, 'pine_tree': 88, 'willow_tree': 89,
    'bicycle': 90, 'bus': 91, 'motorcycle': 92, 'pickup_truck': 93, 'train': 94,
    'lawn_mower': 95, 'rocket': 96, 'streetcar': 97, 'tank': 98, 'tractor': 99
}


def cifar100_train_set(data_root='./datasets'):
    """得到 CIFAR-100 训练集 Dataset, 按照一般惯例进行数据增强"""
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
    """得到 CIFAR-100 测试集 Dataset"""
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
