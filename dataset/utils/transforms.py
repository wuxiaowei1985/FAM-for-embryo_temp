import random
import torchvision.transforms.functional as TF
import torch

class FocusTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size
    def __call__(self, images):
        output = []
        for img in images:
            if random.random() < 0.5:
                img = TF.hflip(img)
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle)
            img = TF.resize(img,[self.image_size, self.image_size])
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5], std=[0.5])
            output.append(img)
        output = torch.stack(output, dim=0)
        return output

class FocusValTransform:   # 验证/测试用（无增强）
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, images):
        output = []
        for img in images:
            # 只做 resize、to_tensor、normalize，不随机翻转/旋转
            img = TF.resize(img, [self.image_size, self.image_size])
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5], std=[0.5])
            output.append(img)
        return torch.stack(output, dim=0)


if __name__ == "__main__":
    ...