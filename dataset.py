import torch.utils.data as data
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        img_transformed = self.transform(
            img, self.phase)  

        if label == "False":
            label = 0
        elif label == "True":
            label = 1

        return img_transformed, label