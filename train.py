import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import NARSILModel
from utilities import (
    TrainMode,
    TSPDataset,
    adj_list_to_tour,
    batched_two_opt,
    edge_list_to_adj_list,
    generate_random_rotation_matrices,
    get_checkpoint,
    load_checkpoint,
    save_checkpoint,
    select_seq_edge,
    self_improvement_learning,
    setup_distributed,
    setup_logging,
    supervised,
    transform_graphs,
)

torch.set_printoptions(precision=3, sci_mode=False)


@torch.inference_mode()
def validate_batch(
    config: dict,
    model: NARSILModel,
    batch_data: list[torch.Tensor],
    device: torch.device,
    rotation_matrix: torch.Tensor = None,
):
    graphs = batch_data["graphs"].to(device)
    cost_matrix = torch.cdist(graphs, graphs, p=2)

    n_unique_graphs, graph_size, _ = graphs.shape

    graphs = transform_graphs(
        graphs=graphs,
        K=config["k_transforms"],
        R=rotation_matrix,
    ).reshape(-1, graph_size, 2)

    # Forward pass through the model
    probs = model(graphs)
    probs: torch.Tensor

    tour_edges = select_seq_edge(probs=probs, cost_matrix=cost_matrix)

    tour_edges: torch.Tensor

    *_, n_seq, graph_size, _ = tour_edges.shape

    tour_adj_list = edge_list_to_adj_list(tour_edges)
    tours = adj_list_to_tour(tour_adj_list)
    two_opt_tours, two_opt_cost, iterations = batched_two_opt(
        tour=tours[:, 0],
        cost_matrix=cost_matrix.unsqueeze(1)
        .expand(-1, config["k_transforms"], -1, -1)
        .reshape(-1, graph_size, graph_size),
        max_iterations=1000,
    )
    two_opt_cost = two_opt_cost.reshape(n_unique_graphs, -1)
    two_opt_tours = two_opt_tours.reshape(n_unique_graphs, -1, graph_size)

    tour_edges = tour_edges.reshape(n_unique_graphs, -1, n_seq, graph_size, 2)
    tour_costs = cost_matrix[
        torch.arange(n_unique_graphs)[:, None, None, None],
        *tour_edges.movedim(-1, 0),
    ].sum(-1)
    # tour_cost = get_selected_cost(tour_matrix=tour_edges, cost_matrix=cost_matrix)
    greedy_cost = tour_costs[..., 0, 0]

    batch_idx = torch.arange(n_unique_graphs)

    best_tour_idx = two_opt_cost.argmin(-1)
    best_cost = two_opt_cost[batch_idx, best_tour_idx]

    cost = torch.stack([greedy_cost, best_cost], dim=-1)

    metrics = {
        "Cost": cost,
        "Greedy is Best": iterations.eq(0).half(),
        "2-Opt Iterations": iterations.float(),
    }
    if "tour_costs" in batch_data:
        target_cost = batch_data["tour_costs"].to(device)

        if cost.ndim == 2:
            target_cost = target_cost.unsqueeze(1)
        metrics["Optimality Gap"] = cost / target_cost - 1

    if dist.is_initialized():
        for tensor in metrics.values():
            if isinstance(tensor, torch.Tensor):
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    return (probs, tours, metrics, graph_size)


@torch.inference_mode()
def validate_epoch(
    config: dict,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    k_transforms = config["k_transforms"]

    rotation_matrix = (
        generate_random_rotation_matrices(
            n=dataloader.batch_size * (k_transforms - 1),
            device=device,
        )
        if k_transforms > 1
        else None
    )
    agg = defaultdict(list)
    for batch_data in dataloader:
        *_, metrics, graph_size = validate_batch(
            config=config,
            batch_data=batch_data,
            model=model,
            device=device,
            rotation_matrix=rotation_matrix,
        )
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):
                agg[metric].append(value)

    metrics = {}
    for metric, values in agg.items():
        factor = 1 if metric in {"Cost", "2-Opt Iterations"} else 100
        value = torch.cat(values, dim=0).float().mean(dim=0) * factor
        match value.numel():
            case 1:
                metrics[metric] = value.item()
            case 2:
                metrics[f"{metric} Greedy"], metrics[f"{metric} Best"] = value.tolist()
            case _:
                raise ValueError(
                    f"Unexpected number of elements in metric '{metric}': {value.numel()}"
                )

    return metrics


# Train one batch
def train_batch(
    config: dict,
    batch_data: dict[str, torch.Tensor],
    model: NARSILModel,
    optimizer: optim.Optimizer,
    device: torch.device,
    rotation_matrix: torch.Tensor = None,
    mode=TrainMode.SELF_IMPROVEMENT,
):
    graphs = batch_data["graphs"].to(device)
    edges = torch.cdist(graphs, graphs, p=2)

    batch_size, graph_size, _ = graphs.shape

    transformed = transform_graphs(
        graphs=graphs,
        K=config["k_transforms"],
        R=rotation_matrix,
    ).reshape(-1, graph_size, 2)

    # Forward pass through the model
    probs = model(transformed)

    match mode:
        case TrainMode.SELF_IMPROVEMENT:
            loss, metrics = self_improvement_learning(
                config=config,
                probs=probs,
                cost_matrix=edges,
            )
        case TrainMode.SUPERVISED:
            assert "tours" in batch_data, (
                "Targets must be provided for supervised training"
            )
            loss, metrics = supervised(
                probs=probs.reshape(batch_size, -1, graph_size, graph_size),
                batch_data=batch_data,
            )
        case _:
            raise ValueError(f"Unknown training mode: {mode}")

    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Compute gradient norm
    total_norm = torch.nn.utils.get_total_norm(
        tensors=[p.grad for p in model.parameters() if p.grad is not None],
        error_if_nonfinite=True,
    )

    # Optionally clip gradients
    if config.get("clip_grads", True):
        torch.nn.utils.clip_grads_with_norm_(
            parameters=model.parameters(), max_norm=1.0, total_norm=total_norm
        )

    # Update model parameters
    optimizer.step()

    metrics["Loss"] = loss
    metrics["Grad Norm"] = total_norm

    for metric, tensor in metrics.items():
        if isinstance(tensor, torch.Tensor):
            if dist.is_initialized():
                dist.all_reduce(tensor.contiguous(), op=dist.ReduceOp.AVG)
            metrics[metric] = (
                tensor.item() if tensor.numel() == 1 else tensor.mean().item()
            )

    return metrics


# Train one epoch
def train_epoch(
    epoch: int,
    config: dict,
    model: NARSILModel,
    optimizer: optim.Optimizer,
    device: torch.device,
    pbar: tqdm,
    best_cost: dict[str, float],
    dataset: TSPDataset = None,
    mode=TrainMode.SELF_IMPROVEMENT,
    scheduler: optim.lr_scheduler.LRScheduler = None,
    val_dataloaders: dict[int, DataLoader] = None,
    writer: SummaryWriter = None,
    rank=0,
):
    model.train()

    general_config = config["general"]
    training_params = config["training"]
    validation_params = config["validation"]

    graph_sizes = training_params["graph_sizes"]
    if len(graph_sizes) == 1:
        idx = 0
    else:  # randomly select a graph size for this epoch
        idx = torch.randint(0, len(graph_sizes), (1,), device=device)

        if dist.is_initialized():  # Synchronize graph sizes across all processes
            dist.broadcast(idx, src=0)

    graph_size = graph_sizes[idx]

    unique_graphs_per_batch = int(
        training_params["nodes_per_batch"]
        // graph_size
        // training_params["k_transforms"]
    )

    # Initialize dataset and dataloader
    if dataset is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        batches_per_rank = max(1, training_params["batches_per_epoch"] // world_size)
        dataset_size = unique_graphs_per_batch * batches_per_rank

        dataset = TSPDataset(graph_size=graph_size, dataset_size=dataset_size)

    sampler = (
        DistributedSampler(dataset=dataset, drop_last=True)
        if dist.is_initialized() and mode == TrainMode.SUPERVISED
        else None
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=unique_graphs_per_batch,
        num_workers=0,  # 0 seems to give best performance
        pin_memory=True,
        shuffle=sampler is None,
        sampler=sampler,
    )

    rotation_matrix = (
        generate_random_rotation_matrices(
            n=unique_graphs_per_batch * (training_params["k_transforms"] - 1),
            device=device,
        )
        if training_params["k_transforms"] > 1
        else None
    )

    for batch_data in dataloader:
        metrics = train_batch(
            config=training_params,
            batch_data=batch_data,
            model=model,
            optimizer=optimizer,
            device=device,
            rotation_matrix=rotation_matrix,
            mode=mode,
        )

        postfix = {metric: f"{value:.3f}" for metric, value in metrics.items()} | {
            "Graph Size": graph_size,
        }

        pbar.set_postfix(postfix)

    if isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        scheduler.step()

    if rank == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict()
            if isinstance(model, DDP)
            else model.state_dict(),
        }

        checkpoint_dir = Path(general_config["outputs_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if epoch % general_config["save_every"] == 0:
            save_checkpoint(
                checkpoint=checkpoint,
                path=checkpoint_dir / "latest_model.pt",
            )

        if (
            isinstance(writer, SummaryWriter)
            and epoch % general_config["log_every"] == 0
        ):
            if graph_size % 10 == 0:
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    writer.add_scalar(
                        tag=f"TSP{graph_size}/Train {key}",
                        scalar_value=value,
                        global_step=epoch,
                    )

            current_lr = (
                scheduler.get_last_lr()[0]
                if isinstance(scheduler, optim.lr_scheduler.LRScheduler)
                else training_params["lr"]
            )
            writer.add_scalar("Learning Rate", current_lr, epoch)

    # Validate the model on a validation set if provided
    if val_dataloaders and (
        epoch == 1 or epoch % validation_params["validate_every"] == 0
    ):
        for val_graph_size, val_dataloader in val_dataloaders.items():
            metrics = validate_epoch(
                config=validation_params,
                model=model,
                dataloader=val_dataloader,
                device=device,
            )
            if rank == 0:
                val_cost_greedy = metrics["Cost Greedy"]
                # If best model, save it
                if val_cost_greedy < best_cost.get(val_graph_size, float("inf")):
                    best_cost[val_graph_size] = val_cost_greedy
                    new_best_path = (
                        checkpoint_dir
                        / f"best_model_tsp{val_graph_size}_{val_cost_greedy:.4f}.pt"
                    )

                    save_checkpoint(checkpoint=checkpoint, path=new_best_path)

                    # Delete any other existing best_model_tsp{val_graph_size}_*.pt files
                    for old_best_path in checkpoint_dir.glob(
                        f"best_model_tsp{val_graph_size}_*.pt"
                    ):
                        old_best_path.samefile(new_best_path) or old_best_path.unlink()

                    timestamp = datetime.now().isoformat(sep=" ", timespec="minutes")
                    pbar.write(
                        f"{timestamp} | Epoch {epoch:>6,} | New best TSP{val_graph_size:>3} model saved! 🚀 Cost: {val_cost_greedy:.4f}"
                    )

                if isinstance(writer, SummaryWriter):
                    for key, value in metrics.items():
                        if not isinstance(value, (int, float)):
                            continue
                        writer.add_scalar(
                            tag=f"TSP{val_graph_size}/Validation {key}",
                            scalar_value=value,
                            global_step=epoch,
                        )

    if dist.is_initialized():  # Synchronize processes before moving to the next epoch
        dist.barrier()


def train(config: dict[str, dict[str]], rank=0, local_rank=0):
    # Set device (CPU or GPU)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Get checkpoint if exists
    checkpoint, checkpoint_config = get_checkpoint(
        path=config["general"].get("load_checkpoint", None),
        device=device,
    )

    # Merge checkpoint config with provided config
    if isinstance(checkpoint_config, dict):
        config = {
            config_group: params | config.get(config_group, {})
            for config_group, params in checkpoint_config.items()
        }

    general_config = config["general"]
    training_params = config["training"]
    validation_params = config["validation"]

    # Save config to run_dir/train.config
    if rank == 0:
        save_config_path = Path(general_config["outputs_dir"]) / "train.config"

        save_config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config_path.write_text(json.dumps(config, indent=4))

        logging.info(f"Config saved to {save_config_path}")

    # Initialize model
    model = NARSILModel(**config["model"]).to(device)
    if dist.is_initialized():
        model = DDP(
            module=model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )

    num_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"Model has {num_params:,} parameters")
    logging.info(f"Model has {num_params:,} parameters")

    # Loss function, optimizer, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=training_params["lr"])

    warmup_scheduler = (
        LinearLR(
            optimizer=optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=training_params["warmup_epochs"],
        )
        if training_params["warmup_epochs"] > 0
        else None
    )
    eta_min_factor = min(0.9, training_params["lr_min_factor"])
    eta_min = training_params["lr"] * eta_min_factor
    match training_params["scheduler"]:
        case "CosineWarmupScheduler":
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=eta_min,
                T_max=training_params["num_epochs"] - training_params["warmup_epochs"],
            )
        case "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                eta_min=eta_min,
                T_0=training_params["lr_restart_every"],
            )
        case "ExponentialLR":
            gamma = eta_min_factor ** (1 / training_params["num_epochs"])
            scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)
        case _:
            scheduler = None

    if isinstance(warmup_scheduler, optim.lr_scheduler.LRScheduler):
        if isinstance(scheduler, optim.lr_scheduler.LRScheduler):
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[training_params["warmup_epochs"]],
            )
        else:
            scheduler = warmup_scheduler
    if isinstance(checkpoint, dict):  # Load checkpoint if exists
        load_checkpoint(checkpoint=checkpoint, model=model)
        logging.info(f"Resumed from checkpoint: {config['general']['load_checkpoint']}")

    val_dataloaders = {}
    if validation_params["validate_every"] > 0:
        for graph_size in validation_params["graph_sizes"]:
            # Initialize validation dataset and dataloader
            batch_size = int(
                validation_params["nodes_per_batch"]
                // graph_size
                // validation_params["k_transforms"]
            )
            dataset_size = batch_size * validation_params["batches_per_epoch"]
            dataset_path = Path(f"dataset/tsp{graph_size}_test_concorde.pt")
            validation_dataset = TSPDataset(
                dataset_path=dataset_path if dataset_path.is_file() else None,
                graph_size=graph_size,
                dataset_size=dataset_size,
            )
            sampler = (
                DistributedSampler(validation_dataset, drop_last=True)
                if dist.is_initialized()
                else None
            )
            validation_dataloader = DataLoader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=0,  # 0 seems to give best performance
                pin_memory=True,
                shuffle=sampler is None,
                sampler=sampler,
            )
            val_dataloaders[graph_size] = validation_dataloader

    graph_sizes = training_params.get("graph_sizes", [])
    assert isinstance(graph_sizes, list) and graph_sizes, (
        "'graph_sizes' must be specified as a non-empty list of integers in the config"
    )

    mode = {
        "self_improvement": TrainMode.SELF_IMPROVEMENT,
        "supervised": TrainMode.SUPERVISED,
    }.get(training_params["mode"], TrainMode.SELF_IMPROVEMENT)

    datasets = (
        {
            graph_size: TSPDataset(
                dataset_path=Path(f"dataset/tsp{graph_size}_train_seed1234.pt")
            )
            for graph_size in set(graph_sizes)
        }
        if mode == TrainMode.SUPERVISED
        else None
    )

    comment = " ".join(
        segment
        for segment in (
            "TSP",
            str(graph_sizes),
            f"({num_params:,})",
            general_config.get("tensorboard_comment", ""),
        )
    )

    # Initialize TensorBoard writer (only for rank 0)
    writer = (
        SummaryWriter(
            log_dir=f"runs/{datetime.now().isoformat(sep=' ', timespec='seconds')} {comment}",
        )
        if rank == 0
        else None
    )

    pbar = tqdm(
        iterable=range(training_params["num_epochs"]),
        desc="Training",
        unit="epoch",
        disable=rank != 0,  # Disable progress bar for non-master ranks
    )

    logging.info(f"Starting training for {training_params['num_epochs']} epochs...")
    logging.info(pformat(config, indent=4))

    best_cost = {}

    # with tqdm_logging_redirect():
    for epoch in pbar:  # Training loop
        dataset = None
        if isinstance(datasets, dict):
            graph_sizes = training_params["graph_sizes"]
            idx = torch.randint(0, len(graph_sizes), (1,), device=device)
            graph_size = graph_sizes[idx]
            indices = torch.randint(
                low=0,
                high=len(datasets[graph_size]),
                size=(
                    int(
                        training_params["nodes_per_batch"]
                        // graph_size
                        // training_params["k_transforms"]
                    )
                    * training_params["batches_per_epoch"],
                ),
                device=device,
            )
            if dist.is_initialized():
                # Synchronize graph sizes across all processes
                dist.broadcast(indices, src=0)
            dataset = torch.utils.data.Subset(
                dataset=datasets[graph_size],
                indices=indices,
            )
        train_epoch(
            epoch=epoch + 1,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pbar=pbar,
            best_cost=best_cost,
            dataset=dataset,
            mode=mode,
            val_dataloaders=val_dataloaders,
            writer=writer,
            rank=rank,
        )

    pbar.close()

    # Close TensorBoard writer
    if isinstance(writer, SummaryWriter):
        writer.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file",
    )
    args = parser.parse_args()

    # Setup distributed environment
    rank, local_rank = setup_distributed()

    # Load configuration
    config_path: Path = args.config
    default_config_path = Path("configs/train.config")
    if isinstance(config_path, Path):
        if config_path.is_file():
            print(f"Loading config from {config_path}")
        else:
            print(f"Config file at {config_path} does not exist. Exiting.")
            return
    else:
        if rank == 0:
            print(
                f"No config file provided. Trying to load config from {default_config_path}"
            )
        if default_config_path.is_file():
            config_path = default_config_path
        else:
            default_config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config_path.touch()
            if rank == 0:
                print(
                    f"Created default config file at {default_config_path}. Please add your configuration there."
                )
            return

    config = json.loads(config_path.read_text())

    setup_logging(rank=rank, output_dir=Path(config["general"]["outputs_dir"]) / "logs")
    # logging.disable()
    logging.info(f"Loaded config from {config_path}")

    train(config=config, rank=rank, local_rank=local_rank)

    if dist.is_initialized():  # Cleanup distributed environment
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
