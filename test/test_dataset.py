import matplotlib.pyplot as plt
from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.transforms import FocusTransform


def test_dataset():
    dataset = EmbryoDataset(root=r"I:\datasets\胚胎\南特704", transform=FocusTransform())
    print(len(dataset))
    sample = dataset[0]
    print(sample["embryo"])
    print(sample["image_name"])
    print(sample["run"])
    print(sample["label"])
    print(sample["images"].shape)

def test_visit():
    dataset = EmbryoDataset(root=r"I:\datasets\胚胎\南特704")
    sample = dataset[0]
    imgs = sample["images"]
    fig, ax = plt.subplots(1, 7, figsize=(18, 3))
    titles = ["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45"]
    for i in range(7):
        ax[i].imshow(imgs[i], cmap="gray")
        ax[i].set_title(titles[i])
        ax[i].axis("off")
    plt.show()

if __name__ == "__main__":
    test_dataset()
    # visit()


