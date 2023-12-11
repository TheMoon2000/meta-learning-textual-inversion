import diffusers
from diffusers import DiffusionPipeline
from diffusers import schedulers, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
import torch
from torch.utils.data import DataLoader
import transformers.models
from ghostfacenetsv2 import GhostFaceNetsV2
from transformers import CLIPTextModel, CLIPTokenizer
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import utils
import cv2, os, math
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import logging
import itertools
import argparse
from tensorboard.summary import Writer
import datetime

writer = Writer("logs/" + datetime.datetime.now().isoformat())

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def create_dataloader(dataset, train_batch_size=1):
    return DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

def train(args):
    text_encoder: transformers.models.clip.modeling_clip.CLIPTextModel = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet"
    )
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )

    placeholder_token = "<xyz>"
    if tokenizer.add_tokens(placeholder_token):
        text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    print("Placeholder token id:", placeholder_token_id)
    
    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    freeze_params(text_encoder.text_model.encoder.parameters())
    freeze_params(text_encoder.text_model.final_layer_norm.parameters())
    freeze_params(text_encoder.text_model.embeddings.position_embedding.parameters())

    train_dataset = utils.TextualInversionDataset('data', tokenizer, placeholder_token=placeholder_token, skip_faceid=args.ghostnet)
    if args.ghostnet:
        embedding_predictor = GhostFaceNetsV2(image_size=178, num_classes=768, dropout=0)
    else:
        embedding_predictor = utils.MetaTextInversion()
    if args.mlp_path:
         embedding_predictor.load_state_dict(torch.load(args.mlp_path), strict=False)
    
    noise_scheduler = PNDMScheduler.from_config(args.model_path, subfolder="scheduler")

    hyperparameters = {
        "learning_rate": 5e-4,
        "scale_lr": True,
        "max_train_steps": args.max_steps,
        "save_steps": args.save_steps,
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 10,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "seed": 42,
        "output_dir": args.output_dir
    }

    os.makedirs(args.output_dir, exist_ok=True)

    logger = logging.getLogger()

    def training_function(text_encoder, vae, unet, global_step=0):
        train_batch_size = hyperparameters["train_batch_size"]
        gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
        learning_rate = hyperparameters["learning_rate"]
        max_train_steps = hyperparameters["max_train_steps"]
        output_dir = hyperparameters["output_dir"]
        gradient_checkpointing = hyperparameters["gradient_checkpointing"]

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        train_dataloader = create_dataloader(train_dataset, train_batch_size)

        if hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            itertools.chain(
                embedding_predictor.parameters(),
                text_encoder.get_input_embeddings().parameters()
            ),  # only optimize the embeddings and the embedding predictor MLP
            lr=learning_rate,
        )

        if args.optimizer_path:
            optimizer.load_state_dict(
                torch.load(args.optimizer_path)
            )

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        embedding_predictor.to(accelerator.device, dtype=weight_dtype)

        # Keep vae in eval mode as we don't train it
        vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        unet.train()

        # Train embedding predictor
        embedding_predictor.train()

        
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        print("Total number of epochs:", num_train_epochs)

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if not args.ghostnet and torch.min(batch['valid']) == 0: continue
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    embeddings = text_encoder.get_input_embeddings().weight.data
                    if args.ghostnet:
                        mapped_embeddings = embedding_predictor(batch["pixel_values"].to(dtype=weight_dtype))
                    else:
                        mapped_embeddings = embedding_predictor(batch["face_id"].to(dtype=weight_dtype))
                    batched_hidden_states = []
                    for b in range(mapped_embeddings.shape[0]):
                        embeddings[placeholder_token_id] = mapped_embeddings[b]
                        encoder_hidden_states = text_encoder(batch["input_ids"][[b]])[0]
                        batched_hidden_states.append(encoder_hidden_states)
                    batched_hidden_states = torch.vstack(batched_hidden_states)

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, batched_hidden_states.to(weight_dtype)).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % hyperparameters["save_steps"] == 0:
                        save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                        # save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
                        torch.save(embedding_predictor.state_dict(), os.path.join(output_dir, f'ghostnet-{global_step}.bin' if args.ghostnet else f'mlp-{global_step}.bin'))
                        optimizer: torch.optim.Optimizer
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer-{global_step}.pt"))

                logs = {"loss": loss.detach().item()}
                writer.add_scalar("Training Loss", loss.detach().item(), step=global_step)
                writer.flush()
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()


        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)
            # Also save the newly trained embeddings
            # save_path = os.path.join(output_dir, f"learned_embeds.bin")
            torch.save(embedding_predictor.state_dict(), os.path.join(output_dir, f'mlp-final.bin'))
            writer.close()
            # save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
        
    accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, args.resume_from), num_processes=1)
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
    torch.cuda.empty_cache()


# pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for meta text-inversion")
    parser.add_argument('--model_path', default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument('--mlp_path', type=str)
    parser.add_argument('--ghostnet', action="store_true")
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--resume_from', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--optimizer_path", type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="meta-text-inversion-output")
    args = parser.parse_args()

    train(args)