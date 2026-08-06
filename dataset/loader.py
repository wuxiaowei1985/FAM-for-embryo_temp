from dataset.embryo_dataset import EmbryoDataset
from dataset.utils.collate_fn import embryo_collate_fn
from dataset.utils.split import split_embryos
from dataset.utils.transforms import FocusTransform
from torch.utils.data import DataLoader
from configs import config as cfg


train_embryos, val_embryos, test_embryos = split_embryos(cfg.DATA_ROOT_A, seed=42)

train_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                              transform=FocusTransform(),
                              embryo_list=train_embryos
                              )
val_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                            transform=FocusTransform(),
                            embryo_list=val_embryos
                            )
test_dataset = EmbryoDataset(root=cfg.DATA_ROOT_A,
                             transform=FocusTransform(),
                             embryo_list=test_embryos
                             )
train_loader = DataLoader(train_dataset,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=True,
                          num_workers=cfg.NUM_WORKERS,
                          collate_fn=embryo_collate_fn
                          )
val_loader = DataLoader(val_dataset,
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=False,
                        num_workers=cfg.NUM_WORKERS,
                        collate_fn=embryo_collate_fn
                        )
test_loader = DataLoader(test_dataset,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=False,
                         num_workers=cfg.NUM_WORKERS,
                         collate_fn=embryo_collate_fn
                         )