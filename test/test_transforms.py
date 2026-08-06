from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.transforms import FocusTransform


def test_transform():
    dataset = EmbryoDataset(root=r"I:\datasets\胚胎\南特704", transform=FocusTransform())
    sample = dataset[0]
    print(sample["images"].shape)

if __name__ == "__main__":
    test_transform()