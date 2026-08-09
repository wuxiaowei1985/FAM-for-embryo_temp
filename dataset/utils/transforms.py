import random
import torchvision.transforms.functional as TF
import torch

class FocusTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size
    def __call__(self, images):
        output = []
        for img in images:
            # 新增：随机调整亮度和对比度（概率 0.5）
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
            # 新增：随机平移 (shift ~ 10%)
            if random.random() < 0.5:
                h, w = img.size
                max_dx = int(w * 0.08)
                max_dy = int(h * 0.08)
                dx = random.randint(-max_dx, max_dx)
                dy = random.randint(-max_dy, max_dy)
                img = TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0)
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