
import os
import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from psychai.vision import load_folders
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from timm.data.transforms_factory import create_transform
from psychai.vision.vm import TrainingManager
from psychai.config import EvaluationConfig, update_config

def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "resnet18",
            "model_type": "resnet",
        },
        "data": {
            "test_path": f"./data/pixel_files",
            "batch_size": 8,
            "data_process_batch_size": 16,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_embeddings": True,
            "layer_of_interest": 0,
        },
        "root_dir": "./",
        "exp_name": "resnet18_evaluation",
        "exp_dir": "./resnet18",
        "task": "feature_extraction",
        "device": "cpu"
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    folder_path = "./data/images"
    jsonl_path = "./data/pixel_files/images_dataset.jsonl"

    # load_folders(
    #     folder_path=folder_path,
    #     jsonl_path=jsonl_path,)
    # return 

    tm = TrainingManager(cfg=cfg)

    tm.mm.load_model(
        model_name=cfg.model.name,
        model_path=cfg.model.path,
        model_type=cfg.model.model_type,
        device=cfg.device,
        task=cfg.task
    )

    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    old_columns = dataset.column_names
    model_transform = create_transform(**tm.mm.model_config, is_training=False)

    def apply_transform(example):
        tensor = torch.tensor(example["pixel_values"])
        img = to_pil_image(tensor)
        example["pixel_values"] = model_transform(img)
        return example
    
    dataset = dataset.map(apply_transform)
    dataset.set_format(type="torch", columns=["pixel_values"])

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    tm.evaluate(
        dataloader=loader,
        eval_fn=None,
        epoch=0,
        step=None, 
    ) 

if __name__ == "__main__":
    main()