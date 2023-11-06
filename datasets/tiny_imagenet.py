"""
Create a tiny imagenet directory of images and class labels
"""

import argparse
import io
import os

from PIL import Image
import numpy as np
from datasets import load_dataset

dataset = load_dataset("zh-plus/tiny-imagenet")

def dump_images(out_dir, images, prefix):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"{prefix}_{i:07d}.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", help="path to output directory")
    args = parser.parse_args()
    out_dir = args.out_dir

    dataset = load_dataset("zh-plus/tiny-imagenet")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, point in enumerate(dataset["train"]):
        image = point["image"]
        label = point["label"]
        image.save(os.path.join(out_dir, f"{label}_{i:07d}.png"))

        #Image.fromarray(image).save(os.path.join(out_dir, f"{label}_{i:07d}.png"))

if __name__ == "__main__":
    main()
