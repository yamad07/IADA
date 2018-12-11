from torchvision.dataset.mnist import MNIST
import random

class DAMNIST(MNIST):
    def __init__(self, root, train=True, source_transform=None, target_transform=None, download=False):
        super(DAMNIST, self).__init__(self, root, train=train, transform=None,
                target_transform=None, download=False)
        self.source_transform = source_transform
        self.target_transform = target_transform

        self.source_data = self.train_data
        print(self.source_data)
        self.source_labels = self.train_labels
        print(self.source_labels)
        self.target_data = self.source_data
        print(self.target_data)

    def __getitem__(self, idx):
        if self.train:
            source_img = self.source_data[idx]
            source_label = self.source_data[idx]
            target_img = self.target_data[idx]

            source_img = Image.fromarray(source_img.numpy(), mode='L')
            target_img = Image.fromarray(target_img.numpy(), mode='L')

            if self.source_transform is not None:
                source_img = self.transform(source_img)

            if self.target_transform is not None:
                target_img = self.transform(target_img)

            return source_img, source_label, target_img

