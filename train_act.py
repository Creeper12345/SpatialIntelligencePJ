"""在 CALVIN LeRobot 数据集上训练 ACT 策略。"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent / "src"))
from calvin_lerobot.dataset import build_calvin_dataset

from calvin_lerobot.act_policy_import import import_act_classes

ACTPolicy, ACTConfig = import_act_classes()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--envs", nargs="+", default=["A"])

    parser.add_argument("--chunk_size",          type=int,   default=10)
    parser.add_argument("--hidden_dim",          type=int,   default=512)
    parser.add_argument("--dim_feedforward",     type=int,   default=3200)
    parser.add_argument("--nheads",              type=int,   default=8)
    parser.add_argument("--num_encoder_layers",  type=int,   default=4)
    parser.add_argument("--num_decoder_layers",  type=int,   default=1)
    parser.add_argument("--latent_dim",          type=int,   default=32)
    parser.add_argument("--kl_weight",           type=float, default=10.0)
    parser.add_argument("--use_wrist_cam",       action="store_true",
                        help="同时使用夹爪视角图像作为额外输入")
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="使用 torchvision ResNet 预训练权重；默认关闭以避免联网下载/坏缓存")
    parser.add_argument("--fps", type=int, default=30,
                        help="CALVIN/LeRobot 数据帧率，用于 action chunk 的时间偏移")

    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="每卡 batch；2×V100-16G 建议 32，32G 建议 64")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="每个 DataLoader worker 预取 batch 数；num_workers=0 时无效")
    parser.add_argument("--val_ratio",    type=float, default=0.05)
    parser.add_argument("--max_steps",    type=int,   default=0,
                        help="调试用：每个 epoch 最多训练多少个 batch；0 表示不限")

    parser.add_argument("--run_name",      type=str, default="act_calvin")
    parser.add_argument("--ckpt_dir",      type=str, default="./checkpoints")
    parser.add_argument("--save_every",    type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="hw3_lerobot_calvin")
    parser.add_argument("--wandb_mode",    type=str, default="online", choices=["online", "offline", "disabled"],
                        help="W&B 模式；无外网机器建议 offline")
    parser.add_argument("--no_wandb",      action="store_true")
    parser.add_argument("--tensorboard_logdir", type=str, default="./tb_logs",
                        help="本地 TensorBoard 日志目录")

    return parser.parse_args()


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, local_rank, global_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main(rank): return rank == 0


def build_act_config(args) -> ACTConfig:
    import inspect

    signature = inspect.signature(ACTConfig)
    params = signature.parameters

    # LeRobot >= 0.4 使用 PolicyFeature 配置。
    if "input_features" in params:
        from lerobot.configs.types import FeatureType, PolicyFeature

        input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        }
        if args.use_wrist_cam:
            input_features["observation.images.wrist_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            )

        return ACTConfig(
            input_features=input_features,
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            chunk_size=args.chunk_size,
            n_action_steps=args.chunk_size,
            dim_model=args.hidden_dim,
            dim_feedforward=args.dim_feedforward,
            n_heads=args.nheads,
            n_encoder_layers=args.num_encoder_layers,
            n_decoder_layers=args.num_decoder_layers,
            latent_dim=args.latent_dim,
            kl_weight=args.kl_weight,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if args.pretrained_backbone else None,
        )

    # LeRobot <= 0.3 使用 shape 字典配置。
    input_shapes = {
        "observation.images.image": [3, 256, 256],
        "observation.state": [15],
    }
    if args.use_wrist_cam:
        input_shapes["observation.images.wrist_image"] = [3, 256, 256]

    return ACTConfig(
        input_shapes=input_shapes,
        output_shapes={"action": [7]},
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
    )


def run_epoch(model, loader, optimizer, device, train: bool, max_steps: int = 0):
    # LeRobot ACT v3 在 train mode 下才返回 VAE latent 统计；验证阶段仍使用 no_grad。
    model.train()
    total_loss = total_l1 = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, start=1):
            if max_steps and step > max_steps:
                break
            obs = {
                "observation.images.image": batch["observation.images.image"]
                    .squeeze(1).to(device),           # (B, 3, 256, 256)
                "observation.state": batch["observation.state"]
                    .squeeze(1).to(device),           # (B, 15)
                "action": batch["action"].to(device), # (B, chunk_size, 7)
            }
            obs["action_is_pad"] = torch.zeros(
                obs["action"].shape[:2], dtype=torch.bool, device=device
            )

            if "observation.images.wrist_image" in batch:
                obs["observation.images.wrist_image"] = (
                    batch["observation.images.wrist_image"].squeeze(1).to(device)
                )

            if train:
                optimizer.zero_grad()

            out = model(obs)
            if isinstance(out, tuple):
                loss, loss_info = out
                l1 = float(loss_info.get("l1_loss", loss.detach().item()))
            else:
                loss = out["loss"]
                value = out.get("l1_loss", loss)
                l1 = value.item() if hasattr(value, "item") else float(value)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            total_loss += loss.item()
            total_l1   += l1
            n += 1

    if n == 0:
        raise RuntimeError("No batches were processed. Check dataset length, batch_size, drop_last, and --max_steps.")
    return total_loss / n, total_l1 / n


def main():
    args = parse_args()
    use_ddp, local_rank, global_rank, world_size = setup_ddp()
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main_p  = is_main(global_rank)

    if main_p:
        print(f"=== ACT on CALVIN ===  envs={args.envs}  world={world_size}  device={device}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
            print(f"GPU[{local_rank}]={torch.cuda.get_device_name(local_rank)}", flush=True)
        print(f"Batch/card={args.batch_size}  Global batch={args.batch_size * world_size}", flush=True)

    tb_writer = None
    if main_p and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(Path(args.tensorboard_logdir) / args.run_name))

    if main_p and not args.no_wandb and args.wandb_mode != "disabled":
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args), mode=args.wandb_mode)

    if main_p:
        print("Building dataset ...", flush=True)

    full_ds = build_calvin_dataset(
        args.data_dir,
        envs=args.envs,
        action_horizon=args.chunk_size,
        fps=args.fps,
        use_wrist_cam=args.use_wrist_cam,
    )

    val_size   = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    if main_p:
        print(f"Train={train_size}  Val={val_size}")

    train_sampler = DistributedSampler(train_ds) if use_ddp else None
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        })

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if main_p:
        print("Building ACT model ...", flush=True)

    model = ACTPolicy(build_act_config(args)).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    if main_p:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        tr_loss, tr_l1 = run_epoch(model, train_loader, optimizer, device, train=True, max_steps=args.max_steps)
        val_max_steps = max(1, args.max_steps // 4) if args.max_steps else 0
        vl_loss, vl_l1 = run_epoch(model, val_loader,   optimizer, device, train=False, max_steps=val_max_steps)
        scheduler.step()

        if main_p:
            elapsed = time.time() - t0
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"tr_loss={tr_loss:.4f} tr_l1={tr_l1:.4f}  "
                f"vl_loss={vl_loss:.4f} vl_l1={vl_l1:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s"
            )

            metrics = {
                "epoch": epoch,
                "train/loss": tr_loss, "train/l1_loss": tr_l1,
                "val/loss":   vl_loss, "val/l1_loss":   vl_l1,
                "lr": scheduler.get_last_lr()[0],
            }
            if tb_writer is not None:
                for key, value in metrics.items():
                    if key != "epoch":
                        tb_writer.add_scalar(key, value, epoch)
                tb_writer.flush()

            if not args.no_wandb and args.wandb_mode != "disabled":
                wandb.log(metrics)

            def state_dict():
                m = model.module if use_ddp else model
                return m.state_dict()

            if vl_l1 < best_val_l1:
                best_val_l1 = vl_l1
                torch.save({"epoch": epoch, "model_state_dict": state_dict(),
                            "val_l1": vl_l1, "args": vars(args)},
                           ckpt_dir / "best_model.pt")
                print(f"  ✓ best saved  val_l1={vl_l1:.4f}")

            if epoch % args.save_every == 0:
                torch.save({"epoch": epoch, "model_state_dict": state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "args": vars(args)},
                           ckpt_dir / f"epoch_{epoch:04d}.pt")

    if main_p:
        print(f"\n=== 完成  best_val_l1={best_val_l1:.4f} ===")
        if tb_writer is not None:
            tb_writer.close()
        if not args.no_wandb and args.wandb_mode != "disabled":
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
