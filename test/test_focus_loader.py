from dataset.utils.focus_loader import FocusLoader
from configs import config as cfg

def test_focus_loader():
    loader = FocusLoader(cfg.DATA_ROOT)
    imgs = loader.load_focus_images(
        "AA83-7",
        "D2013.01.28_S0717_I132_WELL7_RUN88.jpeg"
    )
    print(imgs)

if __name__ == "__main__":
    test_focus_loader()