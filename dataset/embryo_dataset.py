from pathlib import Path
from torch.utils.data import Dataset
import re
from dataset.utils.focus_loader import FocusLoader
from dataset.annotation_reader import AnnotationLoader

pattern = re.compile(r"RUN(\d+)\.jpeg$")

class EmbryoDataset(Dataset):
    def __init__(self, root, transform=None, embryo_list=None):
        self.root = Path(root)
        self.transform = transform
        self.loader = FocusLoader(root)
        self.annotation = AnnotationLoader(self.root / "embryo_dataset_annotations")
        self.samples = []
        embryo_root = self.root / "embryo_dataset"
        if embryo_list is None:
            embryo_dirs = [p for p in embryo_root.iterdir() if p.is_dir()]
        else:
            embryo_dirs = [embryo_root / e for e in embryo_list]
        # embryo_dirs = embryo_dirs[:10]   #取前十个胚胎
        for embryo_dir in embryo_dirs:
            embryo_name = embryo_dir.name
            for img in sorted(embryo_dir.glob("*.jpeg")):
                run_id = int(pattern.search(img.name).group(1))
                label = self.annotation.get_label(embryo_name, run_id)
                if label < 0:
                    continue
                self.samples.append({
                    "embryo": embryo_name,
                    "image_name": img.name,
                    "run": run_id,
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = self.loader.load_focus_images(sample["embryo"], sample["image_name"])
        if self.transform is not None:
            imgs = self.transform(imgs)
        return {
            **sample,
            "images": imgs
        }



if __name__ == "__main__":
    ...
