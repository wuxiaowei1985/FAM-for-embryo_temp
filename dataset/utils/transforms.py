import random
import torchvision.transforms.functional as TF
import torch

flip = random.random() < 0.5
angle = random.uniform(-10,10)

class FocusTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size
    def __call__(self, images):
        output = []
        for img in images:
            if flip:
                img = TF.hflip(img)
            img = TF.rotate(img, angle)
            img = TF.resize(img,[self.image_size, self.image_size])
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5], std=[0.5])
            output.append(img)
        output = torch.stack(output, dim=0)
        return output

if __name__ == "__main__":
    ...