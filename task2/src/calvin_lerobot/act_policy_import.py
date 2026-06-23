"""按需导入 LeRobot ACT，避免加载全部 policy。

部分 LeRobot 版本会在 ``lerobot.policies.__init__`` 中导入所有 policy。
当前 Python 3.12 环境会先在 Groot 相关 dataclass 处报错，导致 ACT 无法加载。
这里补一个最小 ``lerobot.policies`` 包桩，使 ``lerobot.policies.act.*`` 可以直接导入。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def _ensure_policies_stub() -> None:
    try:
        import lerobot
    except ImportError:
        return

    lerobot_dir = Path(lerobot.__file__).resolve().parent
    policies_dir = lerobot_dir / "policies"
    act_dir = policies_dir / "act"
    if not act_dir.exists():
        return

    existing = sys.modules.get("lerobot.policies")
    if existing is not None and getattr(existing, "__path__", None):
        return

    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(policies_dir)]
    pkg.__file__ = str(policies_dir / "__init__.py")
    pkg.__package__ = "lerobot.policies"
    pkg.__spec__ = importlib.util.spec_from_loader("lerobot.policies", loader=None, is_package=True)
    sys.modules["lerobot.policies"] = pkg


def import_act_classes():
    _ensure_policies_stub()
    try:
        modeling = importlib.import_module("lerobot.policies.act.modeling_act")
        config = importlib.import_module("lerobot.policies.act.configuration_act")
        return modeling.ACTPolicy, config.ACTConfig
    except ImportError:
        modeling = importlib.import_module("lerobot.common.policies.act.modeling_act")
        config = importlib.import_module("lerobot.common.policies.act.configuration_act")
        return modeling.ACTPolicy, config.ACTConfig
