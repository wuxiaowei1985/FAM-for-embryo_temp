import torch

def embryo_collate_fn(batch):
    images = torch.stack([sample["images"] for sample in batch], dim=0)
    embryos = [sample["embryo"] for sample in batch]
    image_names = [sample["image_name"] for sample in batch]
    output = {
        "images": images,
        "embryo": embryos,
        "image_name": image_names
    }
    # 如果以后Dataset增加label
    if "label" in batch[0]:
        output["label"] = torch.tensor([sample["label"] for sample in batch] ,dtype=torch.long)
    # 如果以后增加run编号
    if "run" in batch[0]:
        output["run"] = torch.tensor([sample["run"] for sample in batch], dtype=torch.long)
    return output