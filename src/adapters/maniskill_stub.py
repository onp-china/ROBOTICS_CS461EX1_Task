from typing import Dict, Tuple


def build_maniskill_stub(config: Dict) -> Tuple[object, object]:
    task_name = config.get("data", {}).get("task_name", "unknown_task")
    raise NotImplementedError(
        f"当前还没有接入真实 ManiSkill baseline 数据，task={task_name}。"
        "后续正式接入时，请在这里实现 demonstration 读取和数据切片。"
    )
