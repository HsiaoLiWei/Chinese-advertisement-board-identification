import numpy as np
import random, cv2
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from config import char_to_index, char_to_index_Only, char_to_index_New
# transforms.Lambda(lambda img: img * 2.0 - 1.0)
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 456
train_transforms = transforms.Compose([ # transforms.Resize((size,size)),  # (250,250)
                                    # transforms.CenterCrop(size),   # 224
                                    # transforms.RandomCrop(size),   # 224
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.RandomRotation(random.randint(1, 10)),       #隨機旋轉
                                    transforms.RandomHorizontalFlip(),   #隨機水平翻轉
                                    # transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

train_Noisy_transforms = transforms.Compose([ # transforms.Resize((size,size)),  # (250,250)
                                    # transforms.CenterCrop(size),   # 224
                                    transforms.RandomResizedCrop((size,size)),   # 224
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.RandomRotation(random.randint(1, 10)),       #隨機旋轉
                                    transforms.RandomHorizontalFlip(),   #隨機水平翻轉
                                    # transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

test_transforms = transforms.Compose([  # transforms.Resize((size,size)),
                                        # transforms.CenterCrop(size),
                                        # transforms.RandomCrop(size),   # 224
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

test_Noisy_transforms = transforms.Compose([  # transforms.Resize((size,size)),
                                        # transforms.CenterCrop(size),
                                        transforms.RandomResizedCrop((size,size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

class TextCharData(Dataset):
    def __init__(self, path, istrain= False, transform=None):
        self.img_base = []
        self.transform = train_transforms if istrain else test_transforms
        self.Noisy_transforms = train_Noisy_transforms if istrain else test_Noisy_transforms
        self.istrain = istrain
        with open(path, 'r', encoding="utf-8-sig") as f:
            self.imgs = f.readlines()
        # self.img_base = np.loadtxt(path, dtype = str)
        for i in self.imgs:
            i = i.split(' ')
            # img = Image.open(i[0]).convert("RGB")
            # img = np.array(img)
            self.img_base += [[i[0], char_to_index_New[i[1].strip()]]]

    def _Mosaic_augmentation(self, image):
        new_img = image.copy()
        h, w, n = image.shape
        size = random.randint(1, 7) #馬賽克大小
        for i in range(size, h - 1 - size, size):
            for j in range(size, w - 1 - size, size):
                i_rand = random.randint(i - size, i)
                j_rand = random.randint(j - size, j)
                new_img[i - size:i + size, j - size:j + size] = image[i_rand, j_rand, :]
        return new_img

    def __getitem__(self, index):
        img, label = self.img_base[index]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        # cv2.imshow("image_ori", img.astype('uint8'))
        if self.istrain:
            if random.randint(0, 1) == 1:
                img = cv2.resize(img, (random.randint(18, 56),random.randint(18, 56)))
                img = cv2.resize(img, (size,size))
            else:
                img = cv2.resize(img, (size,size))
            if random.randint(0, 1) == 1:
                img = self._Mosaic_augmentation(img)
        else:
            img = cv2.resize(img, (size,size))
        # cv2.imshow("image_new", img.astype('uint8'))
        # cv2.waitKey() 
        # cv2.destroyAllWindows()
        img = Image.fromarray(np.uint8(img))
        label = int(label)

        if label == char_to_index_New["###"]:
            img = self.Noisy_transforms(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_base)

class TextClassData(Dataset):
    def __init__(self, path, istrain= False, transform=None):
        print("Input size: %d x %d"%(size, size))
        self.img_base = []
        self.transform = train_transforms if istrain else test_transforms
        self.Noisy_transforms = train_Noisy_transforms if istrain else test_Noisy_transforms
        self.istrain = istrain
        with open(path, 'r', encoding="utf-8-sig") as f:
            self.imgs = f.readlines()
        # self.img_base = np.loadtxt(path, dtype = str)
        for i in self.imgs:
            i = i.split(' ')
            # img = Image.open(i[0]).convert("RGB")
            # img = np.array(img)
            self.img_base += [[i[0], char_to_index[i[1].strip()]]]

    def _Mosaic_augmentation(self, image):
        new_img = image.copy()
        h, w, n = image.shape
        size = random.randint(1, 7) #馬賽克大小
        for i in range(size, h - 1 - size, size):
            for j in range(size, w - 1 - size, size):
                i_rand = random.randint(i - size, i)
                j_rand = random.randint(j - size, j)
                new_img[i - size:i + size, j - size:j + size] = image[i_rand, j_rand, :]
        return new_img

    def __getitem__(self, index):
        img, label = self.img_base[index]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (size, size))
        # cv2.imshow("image_ori", img.astype('uint8'))
        if self.istrain:
            img = self._Mosaic_augmentation(img)
        # cv2.imshow("image_new", img.astype('uint8'))
        # cv2.waitKey() 
        # cv2.destroyAllWindows()
        img = Image.fromarray(np.uint8(img))
        label = int(label)

        if label == char_to_index["###"]:
            img = self.Noisy_transforms(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_base)
