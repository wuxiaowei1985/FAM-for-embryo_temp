from dataset.utils.focus_loader import FocusLoader

def test_focus_loader():
    loader = FocusLoader(r"I:\datasets\胚胎\南特704")
    imgs = loader.load_focus_images(
        "AA83-7",
        "D2013.01.28_S0717_I132_WELL7_RUN88.jpeg"
    )
    print(imgs)

if __name__ == "__main__":
    test_focus_loader()