from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils import data

# 导入数据集
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomRotation(15, ), # 在+-15°范围内对图像进行随机旋转
        transforms.RandomCrop(224), # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 以0.5概率对图像进行水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
}

categories = {
    'accordion': 0, 'airplanes': 1, 'anchor': 2, 'ant': 3, 'yin_yang': 4, 'barrel': 5,
    'bass': 6, 'beaver': 7, 'binocular': 8, 'bonsai': 9, 'brain': 10, 'brontosaurus': 11,
    'buddha': 12, 'butterfly': 13, 'camera': 14, 'cannon': 15, 'car_side': 16, 'ceiling_fan': 17,
    'cellphone': 18, 'chair': 19, 'chandelier': 20, 'cougar_body': 21, 'cougar_face': 22,
    'crab': 23, 'crayfish': 24, 'crocodile': 25, 'crocodile_head': 26, 'cup': 27, 'dalmatian': 28,
    'dollar_bill': 29, 'dolphin': 30, 'dragonfly': 31, 'electric_guitar': 32, 'elephant': 33,
    'emu': 34, 'euphonium': 35, 'ewer': 36, 'Faces': 37, 'Faces_easy': 38, 'ferry': 39, 'flamingo': 40,
    'flamingo_head': 41, 'garfield': 42, 'gerenuk': 43, 'gramophone': 44, 'grand_piano': 45,
    'hawksbill': 46, 'headphone': 47, 'hedgehog': 48, 'helicopter': 49, 'ibis': 50, 'inline_skate': 51,
    'joshua_tree': 52, 'kangaroo': 53, 'ketch': 54, 'lamp': 55, 'laptop': 56, 'Leopards': 57, 'wrench': 58,
    'llama': 59, 'lobster': 60, 'lotus': 61, 'mandolin': 62, 'mayfly': 63, 'menorah': 64, 'metronome': 65,
    'minaret': 66, 'Motorbikes': 67, 'nautilus': 68, 'octopus': 69, 'okapi': 70, 'pagoda': 71, 'panda': 72,
    'pigeon': 73, 'pizza': 74, 'platypus': 75, 'pyramid': 76, 'revolver': 77, 'rhino': 78, 'rooster': 79,
    'saxophone': 80, 'schooner': 81, 'scissors': 82, 'scorpion': 83, 'sea_horse': 84, 'snoopy': 85,
    'soccer_ball': 86, 'stapler': 87, 'starfish': 88, 'stegosaurus': 89, 'stop_sign': 90, 'strawberry': 91,
    'sunflower': 92, 'tick': 93, 'trilobite': 94, 'umbrella': 95, 'watch': 96, 'water_lilly': 97, 'wheelchair': 98,
    'wild_cat': 99, 'windsor_chair': 100
}


# 自定义数据集
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.image = []
        for idx in range(len(self.img_labels)):
            img = Image.open(self.img_labels.iloc[idx, 0])
            # 若为单通道图像，则复制为三通道图像（灰度图像）
            if len(img.split()) == 1:
                r, g, b = img, img, img
                img = Image.merge("RGB", (r, g, b))
            tmp = self.transform(img)
            self.image.append(tmp)
        for i in range(len(self.img_labels)):
            self.img_labels.iloc[i, 1] = categories[self.img_labels.iloc[i, 1]]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.img_labels.iloc[idx, 1]
        # print(label)
        return image, label

train_dataset = CustomImageDataset(annotations_file="./train_annotations.csv",
                                   transform=data_transforms['train'])
test_dataset = CustomImageDataset(annotations_file="./test_annotations.csv",
                                  transform=data_transforms['test'])

def load_data_caltech101(batch_size):
    return (data.DataLoader(train_dataset, batch_size, shuffle=True),
            data.DataLoader(test_dataset, batch_size, shuffle=False))