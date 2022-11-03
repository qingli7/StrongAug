from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100


def mnist_data_loader(mnist_folder='./data/mnist_data', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 之后
    ])
    
    train_set = MNIST(mnist_folder, train=True, download=True, transform=transform)
    train_set = MNIST(mnist_folder, train=True, download=True, transform=transform)
    test_set = MNIST(mnist_folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True) #, num_workers=4)
    return  train_loader, test_loader


def fashion_mnist_data_loader(folder='./data/fashion_mnist_data', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = FashionMNIST(folder, train=True, download=True, transform=transform)
    test_set = FashionMNIST(folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def cifar10_data_loader(folder='./data/cifar10_data', batch_size=64, transform=True):
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.24705882352941178), # 随机改变图像的亮度对比度和饱和度
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((32, 32), padding=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]), 
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]), 
    ])
    train_set = CIFAR10(folder, train=True, download=True, transform=train_transform)
    test_set = CIFAR10(folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def cifar100_data_loader(folder='./data/cifar100_data', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = CIFAR100(folder, train=True, download=True, transform=transform)
    test_set = CIFAR100(folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


if __name__ == '__main__':
    data_train, data_test = mnist_data_loader()  # checked
    data_train, data_test = fashion_mnist_data_loader()  # checked
    data_train, data_test = cifar10_data_loader()  # checked
    data_train, data_test = cifar100_data_loader()  # checked
    for x, y in data_train:
        print(x.shape, y.shape, x.max(), x.min())