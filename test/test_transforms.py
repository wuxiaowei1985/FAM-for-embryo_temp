from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.transforms import FocusTransform
from configs import config as cfg

def test_transform():
    dataset = EmbryoDataset(root=cfg.DATA_ROOT, transform=FocusTransform())
    sample = dataset[0]
    print(sample["images"].shape)

if __name__ == "__main__":
    test_transform()