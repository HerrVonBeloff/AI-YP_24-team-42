import torchvision.transforms as transforms
from torch.utils.data import Dataset

class TransformDataset(Dataset):
    def __init__(self, df, new_size):
        self.df = df
        self.new_size = new_size
        self.transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): # без len загрузчик не работает
        return len(self.df)

    def __getitem__(self, idx):
        
        image = self.df[idx]['image']  # загружаем изображение и текст по индексу
        # text = self.df[idx]['text']  текст пока не используем

        image = image.convert('RGB')  # преобразуем в трехканальное
        image = self.transform(image)  # применяем трансформации

        return image # тут были проблемы с типом, но, кажется, сейчас он возвращает тензор