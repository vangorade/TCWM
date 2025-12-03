# TCWM: Temporally Consistent Object-Centric World Models

**Temporally Consistent World Models (TCWM)** is a model-based reinforcement learning algorithm that addresses the critical problem of slot identity swaps in object-centric world models. By integrating an action-conditioned slot-contrastive loss, TCWM enforces temporally consistent object-centric latent dynamics through mutual information maximization, aligning slot representations across consecutive timesteps and actions.

## Key Contributions

- **Temporal Consistency**: Solves slot identity swap problems through action-conditioned slot-contrastive loss
- **Mutual Information Maximization**: Aligns slot representations across consecutive timesteps and actions
- **Stable Object Identity**: Ensures consistent object representation throughout observed and imagined rollouts
- **Improved Planning**: Enables reliable long-horizon imagination and policy learning
- **Task-Agnostic Learning**: Maintains pixel-to-action learning paradigm with principled temporal consistency constraints

## Key Features

- **Object-Centric Representation**: Uses slot-based attention mechanisms to decompose scenes into individual objects
- **Temporal Stability**: Prevents slot identity swaps that undermine long-horizon planning and object-based reasoning
- **Structured Latent Dynamics**: Learns predictable world models with stable object-specific transition functions
- **Enhanced Interpretability**: Provides stable and interpretable attention patterns for complex manipulation tasks

## Installation
### Conda
Start by installing the [multi-object-fetch](https://github.com/maltemosbach/multi-object-fetch) environment suite.
Then add the TCWM dependencies to the conda environment by running:
```bash
conda env update -n mof -f apptainer/environment.yml
```

### Apptainer
Alternatively, we provide an [`Apptainer`](apptainer/multi_object_fetch.def) build file to simplify installation.
To build the `.sif` image, run:
```bash
cd apptainer && apptainer build tcwm.sif multi_object_fetch.def
```
To start a training run inside the container:
```bash
apptainer run --nv ../tcwm.sif python train_sold.py
```
> [!NOTE] 
>If you're on a SLURM cluster, you can submit training jobs using this container with the provided run script `sbatch slurm.sh train_sold.py`.



## Training

The training routine consists of two stages: [pre-training a SAVi model](#pre-training-a-savi-model) and 
[training a TCWM model](#training-a-tcwm-model) with temporal consistency constraints on top of it.

### Pre-training a SAVi model
The SAVi models (or autoencoders generally) are pre-trained on static datasets of random trajectories to learn object-centric representations. 
Such datasets can be generated using the following script:
```bash
python generate_dataset.py experiment=my_dataset env.name=ReachRed_0to4Distractors_Dense-v1
```

To train a SAVi model, specify the dataset to be trained on and model parameters such as the number of slots in [`train_autoencoder.yaml`](./configs/train_autoencoder.yaml) and run:
```bash
python train_autoencoder.py experiment=my_savi_model
```

<details>
    <summary><i>Show sample pre-training results</i></summary>
    Good SAVi models should learn to split the scene into meaningful objects and keep slots assigned to the same object over time.
    Examples of SAVi models pre-trained for a reaching and picking task demonstrate object-centric representations.
</details>

### Training a TCWM model

To train TCWM, a checkpoint path to the pre-trained SAVi model is required, which can be specified in the [`train_sold.yaml`](./configs/train_sold.yaml) configuration file. The TCWM training adds the action-conditioned slot-contrastive loss to enforce temporal consistency.

Then, to start the training, run:
```bash
python train_sold.py
```
All results are stored in the [`experiments`](./experiments) directory.



For further evaluation of a trained model or a set of models in a directory, you can run:
```bash
python evaluate_sold.py checkpoint_path=PATH_TO_CHECKPOINT(S)
```
which will log performance metrics and visualizations for the given checkpoints.

## Checkpoints
Pre-trained SAVi and TCWM models are available in the [`checkpoints`](./checkpoints) directory.
The SAVi checkpoints can be used to begin training TCWM models right away.
Each checkpoint also includes corresponding TensorBoard logs, allowing you to visualize the expected training dynamics:
```bash
tensorboard --logdir checkpoints
```


## Citation
This work introduces TCWM, building upon slot-attention based world models. The paper is currently under review at ICLR 2026. If you find this work useful, please consider citing:

```bibtex
@inproceedings{tcwm2026,
  title={TCWM: Temporally Consistent Object-Centric World Models},
  author={Anonymous Authors},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

This work also builds upon foundational slot-attention and object-centric learning:
```bibtex
@inproceedings{sold2025mosbach,
  title={SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels},
  author={Malte Mosbach and Jan Niklas Ewertz and Angel Villar-Corrales and Sven Behnke},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
