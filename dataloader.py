import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def dataloader(batch_size,
               data_root,
               num_workers):
    transform_train = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader