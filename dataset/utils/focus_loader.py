from pathlib import Path
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

FOCUS_ORDER = ["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45",]

ROOT_MAP = {
    "F0": "embryo_dataset",
    "F15": "embryo_dataset_F15",
    "F30": "embryo_dataset_F30",
    "F45": "embryo_dataset_F45",
    "F-15": "embryo_dataset_F-15",
    "F-30": "embryo_dataset_F-30",
    "F-45": "embryo_dataset_F-45",
}
class FocusLoader:
    def __init__(self, root):
        self.root = Path(root)

    def load_focus_images(self, embryo_name, image_name):
        imgs = []
        for focus in FOCUS_ORDER:
            img_path = self.root / ROOT_MAP[focus] / embryo_name / image_name
            try:
                # 用 with 自动关闭文件
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    # copy 到内存，避免文件关闭后引用失效
                    imgs.append(img.copy())
            except Exception as e:
                print(f"\nError Image:\n{img_path}")
                raise RuntimeError(f"Cannot read image: {img_path}") from e
        return imgs

if __name__ == "__main__":
    ...


