import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class FocusTransform:
    def __init__(self, image_size=224, num_views=2, rotation=10.0, translate=0.03, scale_range=(0.97, 1.03), brightness_range=(0.90, 1.10), contrast_range=(0.90, 1.10), noise_std=0.01):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        if num_views < 1:
            raise ValueError("num_views must be >= 1")
        self.image_size = image_size
        self.num_views = num_views
        self.rotation = rotation
        self.translate = translate
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def _sample_params(self):
        angle = random.uniform(-self.rotation, self.rotation)
        max_dx = int(self.image_size * self.translate)
        max_dy = int(self.image_size * self.translate)
        translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
        scale = random.uniform(*self.scale_range)
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        return angle, translate, scale, hflip, vflip, brightness, contrast

    def _transform_stack(self, images, params):
        angle, translate, scale, hflip, vflip, brightness, contrast = params
        output = []
        for img in images:
            img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            if hflip:
                img = TF.hflip(img)
            if vflip:
                img = TF.vflip(img)
            img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            output.append(TF.to_tensor(img))
        tensor = torch.stack(output, dim=0)
        if self.noise_std > 0:
            tensor = tensor + torch.randn_like(tensor) * self.noise_std
        tensor = tensor.clamp(0.0, 1.0)
        tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
        return tensor

    def __call__(self, images):
        if len(images) != 7:
            raise ValueError(f"Expected 7 focal planes, got {len(images)}")
        width, height = images[0].size
        if (width, height) != (500, 500):
            raise ValueError(f"Expected native 500x500 images, "f"got {(width, height)}")
        views = []
        for _ in range(self.num_views):
            params = self._sample_params()
            views.append(self._transform_stack(images, params))
        return torch.stack(views, dim=0)

class FocusValTransform:
    def __init__(self, image_size=224):
        if image_size < 128:
            raise ValueError("image_size must be >= 128.")
        self.image_size = image_size

    def __call__(self, images):
        if len(images) != 7:
            raise ValueError(f"Expected 7 focal planes, got {len(images)}")
        output = []
        for img in images:
            if img.size != (500, 500):
                raise ValueError(f"Expected 500x500 image, got {img.size}")
            # -------------------------------------------------
            # Validation / Test 同样统一 resize
            # -------------------------------------------------
            img = TF.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
            output.append(tensor)
        # [1,7,1,H,W]
        return torch.stack(output, dim=0).unsqueeze(0)