"""在未见过的 CALVIN 环境上评估 ACT 策略。"""

import argparse
from argparse import Namespace
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))
from calvin_lerobot.dataset import build_calvin_dataset, ENV_TO_SUBSET
from train_act import build_act_config

from calvin_lerobot.act_policy_import import import_act_classes

ACTPolicy, ACTConfig = import_act_classes()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--single_ckpt", type=str, required=True,
                        help="单环境模型权重路径（env A only）")
    parser.add_argument("--multi_ckpt",  type=str, required=True,
                        help="多环境模型权重路径（env A+B+C）")
    parser.add_argument("--eval_env",  type=str, default="D",
                        help="Zero-shot 评估的目标环境")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="调试用：最多评估多少个 batch；0 表示全量")
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--run_simulation", action="store_true",
                        help="是否启动 CALVIN 仿真器计算 Success Rate（需安装 calvin-env）")
    parser.add_argument("--num_sim_episodes", type=int, default=100,
                        help="仿真评估的 episode 数量")
    parser.add_argument("--out_path", type=str, default=None,
                        help="结果 JSON 保存路径；默认 logs/eval_env{eval_env}_results.json")
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> ACTPolicy:
    """从 checkpoint 恢复 ACTPolicy"""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]

    config_args = Namespace(
        chunk_size=saved_args["chunk_size"],
        hidden_dim=saved_args["hidden_dim"],
        dim_feedforward=saved_args["dim_feedforward"],
        nheads=saved_args["nheads"],
        num_encoder_layers=saved_args["num_encoder_layers"],
        num_decoder_layers=saved_args["num_decoder_layers"],
        latent_dim=saved_args["latent_dim"],
        kl_weight=saved_args["kl_weight"],
        use_wrist_cam=saved_args.get("use_wrist_cam", False),
        # 评估时 checkpoint 已包含训练后的 backbone 权重。
        # 若保留 pretrained_backbone=True，torchvision 会在 load_state_dict 前尝试下载 ImageNet 权重。
        pretrained_backbone=False,
    )
    config = build_act_config(config_args)
    model = ACTPolicy(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def has_input_feature(model: ACTPolicy, name: str) -> bool:
    config = model.config
    if hasattr(config, "input_features"):
        return name in config.input_features
    if hasattr(config, "input_shapes"):
        return name in config.input_shapes
    return False


@torch.no_grad()
def eval_action_l1(model: ACTPolicy, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict:
    """计算数据集级 Action L1 误差。"""
    all_l1 = []

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break
        obs_image = batch["observation.images.image"].squeeze(1).to(device)
        obs_state = batch["observation.state"].squeeze(1).to(device)
        action_gt = batch["action"].to(device)   # (B, T_act, 7)

        batch_input = {
            "observation.images.image": obs_image,
            "observation.state": obs_state,
        }

        # 数据集级 L1 使用完整 action chunk；新版 LeRobot 的 select_action 可能只返回单步动作。
        if hasattr(model, "predict_action_chunk"):
            pred_actions = model.predict_action_chunk(batch_input)
        else:
            pred_actions = model.select_action(batch_input)
            if pred_actions.ndim == 2:
                pred_actions = pred_actions.unsqueeze(1)

        min_len = min(pred_actions.shape[1], action_gt.shape[1])
        l1 = torch.abs(
            pred_actions[:, :min_len, :] - action_gt[:, :min_len, :]
        ).mean(dim=1)  # (B, 7)

        all_l1.append(l1.cpu().numpy())

    all_l1 = np.concatenate(all_l1, axis=0)   # (N, 7)
    return {
        "l1_mean":    float(all_l1.mean()),
        "l1_per_dim": all_l1.mean(axis=0),    # shape (7,)
    }


def eval_success_rate_simulation(
    model: ACTPolicy,
    data_dir: str,
    eval_env: str,
    num_episodes: int,
    device: torch.device,
    img_size: tuple,
) -> float:
    """运行简化版仿真成功率评估。"""
    try:
        from calvin_env.envs.play_table_env import get_env
        from torchvision import transforms
    except ImportError:
        print("[WARNING] calvin-env 未安装，跳过仿真评估。")
        return -1.0

    env = get_env(
        Path(data_dir) / f"validation/env_{eval_env}",
        show_gui=False,
    )

    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    chunk_size = model.config.chunk_size
    successes = 0

    for ep_idx in range(num_episodes):
        obs = env.reset()
        done = False
        success_in_ep = False
        step = 0
        action_buffer = []

        while not done and step < 360:  # 最多 6 秒 @ 60Hz
            if len(action_buffer) == 0:
                img = img_transform(obs["rgb_obs"]["rgb_static"]).unsqueeze(0).to(device)
                state = torch.tensor(
                    obs["robot_obs"], dtype=torch.float32
                ).unsqueeze(0).to(device)

                batch_input = {
                    "observation.images.image": img,
                    "observation.state": state,
                }
                with torch.no_grad():
                    pred = model.select_action(batch_input)
                    if pred.ndim == 2:
                        pred = pred.unsqueeze(1)
                action_buffer = pred[0].cpu().numpy().tolist()

            action = action_buffer.pop(0)
            obs, reward, done, info = env.step(action)

            if info.get("success", False):
                success_in_ep = True

            step += 1

        if success_in_ep:
            successes += 1

        if (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}/{num_episodes}  "
                  f"running_success_rate={successes/(ep_idx+1):.3f}")

    env.close()
    return successes / num_episodes


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n加载单环境模型: {args.single_ckpt}")
    single_model = load_model(args.single_ckpt, device)

    print(f"加载多环境模型: {args.multi_ckpt}")
    multi_model = load_model(args.multi_ckpt, device)

    print(f"\n加载 CALVIN 环境 {args.eval_env} 验证集 ...")
    eval_ds = build_calvin_dataset(
        data_dir=args.data_dir,
        envs=[args.eval_env],
        split="validation",
        obs_horizon=1,
        action_horizon=single_model.config.chunk_size,
        fps=30,
        use_wrist_cam=has_input_feature(single_model, "observation.images.wrist_image"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"评估样本数: {len(eval_ds)}")

    print(f"\n===== Action L1 Error（环境 {args.eval_env}，Zero-shot）=====")

    single_l1 = eval_action_l1(single_model, eval_loader, device, max_batches=args.max_batches)
    multi_l1  = eval_action_l1(multi_model,  eval_loader, device, max_batches=args.max_batches)

    DIM_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz", "gripper"]
    print(f"\n{'维度':<10} {'单环境(A)':<15} {'多环境(ABC)':<15}")
    print("-" * 40)
    for i, name in enumerate(DIM_NAMES):
        print(f"{name:<10} {single_l1['l1_per_dim'][i]:<15.4f} {multi_l1['l1_per_dim'][i]:<15.4f}")
    print("-" * 40)
    print(f"{'平均':<10} {single_l1['l1_mean']:<15.4f} {multi_l1['l1_mean']:<15.4f}")

    improvement = (single_l1['l1_mean'] - multi_l1['l1_mean']) / single_l1['l1_mean'] * 100
    print(f"\n多环境模型 L1 改善: {improvement:+.1f}%")

    if args.run_simulation:
        print(f"\n===== CALVIN 仿真 Success Rate（环境 {args.eval_env}）=====")
        print(f"评估 episode 数: {args.num_sim_episodes}")

        print("\n[单环境模型]")
        single_sr = eval_success_rate_simulation(
            single_model, args.data_dir, args.eval_env,
            args.num_sim_episodes, device, (args.img_h, args.img_w)
        )

        print("\n[多环境模型]")
        multi_sr = eval_success_rate_simulation(
            multi_model, args.data_dir, args.eval_env,
            args.num_sim_episodes, device, (args.img_h, args.img_w)
        )

        print(f"\n===== Success Rate 汇总 =====")
        print(f"单环境模型 (A only)  : {single_sr:.3f}")
        print(f"多环境模型 (A+B+C)   : {multi_sr:.3f}")
        print(f"改善: {(multi_sr - single_sr)*100:+.1f} pp")

    results = {
        "eval_env": args.eval_env,
        "single_env": {
            "checkpoint": args.single_ckpt,
            "l1_mean": single_l1["l1_mean"],
            "l1_per_dim": single_l1["l1_per_dim"].tolist(),
        },
        "multi_env": {
            "checkpoint": args.multi_ckpt,
            "l1_mean": multi_l1["l1_mean"],
            "l1_per_dim": multi_l1["l1_per_dim"].tolist(),
        },
    }

    import json
    out_path = Path(args.out_path) if args.out_path else Path("./logs") / f"eval_env{args.eval_env}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
