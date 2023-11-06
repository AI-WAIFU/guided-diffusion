"""
Convert images in input_dir to their corresponding latent samples in output_dir with a given model
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from PIL import Image

from scipy.ndimage import zoom

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def get_paths(batch_size, directory_path):
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    total_images = len(image_paths)
    master_index = 0

    while master_index * batch_size * world_size < total_images:
        batch = []
        for i in range(batch_size):
            # Calculate the global index for this file based on the rank and batch size
            index = master_index * batch_size * world_size + i * world_size + rank

            # Break if the index goes out of bounds
            if index >= total_images:
                break

            batch.append(image_paths[index])

        yield batch  # Yield the batch for this rank
        master_index += 1  # Increment the master index to move to the next set of batches

def load_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    np_array = zoom(np.array(pil_image, dtype=np.float32), (4, 4, 1), order=1)
    array = th.FloatTensor(np_array) / 127.5 - 1
    return  array

def dump_sample(sample, path):
    sample_np = sample.cpu().numpy().astype(np.uint8)  # Convert the tensor to a NumPy array
    sample_pil = Image.fromarray(sample_np, 'RGB')  # Create a PIL Image from NumPy array
    sample_pil.save(path)  # Save the image using PIL

def change_path_dir(directory, path):
    filename = os.path.basename(path)  # Extract the filename from the path
    new_path = os.path.join(directory, filename)  # Join the new directory with the filename
    return new_path

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("beggining conversion...")

    for in_path_list in get_paths(args.batch_size, args.in_dir):
        image_list = list(map(load_image, in_path_list))
        image_batch = th.stack(image_list).to(dist_util.dev()).permute(0,3,1,2)

        model_kwargs = {}
        samples = diffusion.ddim_unsample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            image=image_batch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        samples = (samples*args.latent_scale + 127.5).clamp(0, 255).to(th.uint8)
        samples = samples.permute(0, 2, 3, 1)
        
        sample_list = th.unbind(samples)
        out_path_list = map(lambda x : change_path_dir(args.out_dir, x), in_path_list)
        for sample,path in zip(sample_list, out_path_list):
            dump_sample(sample, path)
    
    dist.barrier()
    logger.log("unsampling conversion complete")

def create_argparser():
    defaults = dict(
        latent_scale=50,
        clip_denoised=False,
        batch_size=1,
        use_ddim=True,
        image_size=64,
        model_path="",
        in_dir="",
        out_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
