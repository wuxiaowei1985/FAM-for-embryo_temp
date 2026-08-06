from torch.utils.data import DataLoader
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.transforms import FocusTransform

def test_dataloader():
    dataset = EmbryoDataset(root=r"I:\datasets\胚胎\南特704", transform=FocusTransform())
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=embryo_collate_fn)
    batch = next(iter(loader))
    print(batch["images"].shape)
    print(batch["embryo"])
    print(batch["image_name"])
    print(batch["run"])
    print(batch["label"])


if __name__ == "__main__":
    test_dataloader()

