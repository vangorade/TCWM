import hydra
import numpy as np
from omegaconf import DictConfig
import os
import torch
from torchvision.transforms import ToPILImage
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from PIL import Image


def save_episode(path: str, cfg: DictConfig) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    def save_image(image) -> None:
        if isinstance(image, torch.Tensor):
            if image.dtype == torch.bfloat16:
                image = image.float()
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                img_np = image.cpu().numpy()
            elif image.ndim == 3 and image.shape[0] in (1, 3):
                img_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = np.array(image)

        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        if cfg.save_format == "png":
            Image.fromarray(img_np).save(os.path.join(path, f"{step_count}.png"))
        elif cfg.save_format == "npz":
            images.append(img_np)
        else:
            raise ValueError(f"Unsupported save format: {cfg.save_format}")

    os.makedirs(path, exist_ok=True)
    env = hydra.utils.instantiate(cfg.env)
    step_count, images, actions, rewards = 0, [], [], []
    obs, done = env.reset(), False
    save_image(obs)

    while not done:
        action = torch.from_numpy(env.action_space.sample().astype(np.float32))
        obs, reward, done, _ = env.step(action)
        step_count += 1
        save_image(obs)
        actions.append(action.cpu().numpy())
        rewards.append(reward)

    episode_dict = {"actions": np.stack(actions), "rewards": np.array(rewards)}
    if cfg.save_format == "npz":
        episode_dict["images"] = np.stack(images)
    np.savez_compressed(os.path.join(path, 'episode.npz'), **episode_dict)
    try:
        env.close()
    except Exception:
        pass


@hydra.main(config_path="../configs", config_name="generate_dataset", version_base=None)
def generate_dataset(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    for split in ["train", "val", "test"]:
        num_episodes = getattr(cfg, f"num_{split}")
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        paths = [os.path.join(output_dir, split, str(episode)) for episode in range(num_episodes)]
        cfgs = [cfg] * len(paths)

        if cfg.num_workers > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=ctx) as executor:
                futures = [executor.submit(save_episode, path, cfg) for path, cfg in zip(paths, cfgs)]
                for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc=split.capitalize()):
                    pass
        else:
            progress = tqdm.tqdm(total=len(paths))
            for path, cfg in zip(paths, cfgs):
                save_episode(path, cfg)
                progress.update(1)


if __name__ == "__main__":
    generate_dataset()
