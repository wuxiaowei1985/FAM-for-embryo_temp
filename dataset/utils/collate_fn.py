import torch

def embryo_collate_fn(batch):
    images = torch.cat([sample["images"] for sample in batch], dim=0)
    views_per_sample = batch[0]["images"].shape[0]
    embryos = []
    image_names = []
    runs = []
    labels = []
    for sample in batch:
        embryos.extend([sample["embryo"]] * views_per_sample)
        image_names.extend([sample["image_name"]] * views_per_sample)
        runs.extend([sample["run"]] * views_per_sample)
        labels.extend([sample["label"]] * views_per_sample)
    return {
        "images": images,
        "embryo": embryos,
        "image_name": image_names,
        "run": torch.tensor(runs, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long)
    }