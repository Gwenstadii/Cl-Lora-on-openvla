"""
Unify language instructions across all episodes for each task.
Replaces the 100+ language variants with a single fixed prompt per task,
eliminating the 'language generalization' variable from CL experiments.

Usage (on server, inside RoboTwin-main/):
  python script/unify_prompts.py

Task → Unified prompt:
  handover_mic    → "Pick up the handheld microphone and hand it over"
  grab_roller     → "Grab the smooth wooden roller with both arms"
  stack_bowls_two → "Stack the small smooth brown-rimmed bowl directly over the smooth bowl with glossy finish"
  open_laptop     → "Raise the lid of the rectangular laptop with hinge"
"""

import json
import os
import glob

TASK_PROMPTS = {
    "handover_mic": "Pick up the handheld microphone and hand it over",
    "grab_roller": "Grab the smooth wooden roller with both arms",
    "stack_bowls_two": "Stack the small smooth brown-rimmed bowl directly over the smooth bowl with glossy finish",
    "open_laptop": "Raise the lid of the rectangular laptop with hinge",
}

DATA_ROOT = "data"


def unify_task(task_name: str, prompt: str) -> None:
    instr_dir = os.path.join(DATA_ROOT, task_name, "demo_clean", "instructions")
    if not os.path.isdir(instr_dir):
        print(f"  [SKIP] {instr_dir} not found")
        return

    # 1. Find all episode JSON files
    files = sorted(glob.glob(os.path.join(instr_dir, "episode*.json")))
    if not files:
        print(f"  [SKIP] no episode files in {instr_dir}")
        return

    # 2. Check format from first file
    with open(files[0], "r") as f:
        sample = json.load(f)

    # 3. Determine structure and overwrite
    overwritten = 0
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Overwrite instruction field(s)
            if "instruction" in data:
                data["instruction"] = prompt
            if "instructions" in data:
                data["instructions"] = [prompt]
            if "seen" in data:
                data["seen"] = [prompt]
            if "unseen" in data:
                data["unseen"] = [prompt]
            # If none of the above, replace the entire dict
            if not any(k in data for k in ["instruction", "instructions", "seen", "unseen"]):
                data = {"instruction": prompt}
        elif isinstance(data, list):
            data = [prompt]
        elif isinstance(data, str):
            data = prompt
        else:
            data = {"instruction": prompt}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        overwritten += 1

    print(f"  {task_name}: unified {overwritten} episodes → \"{prompt}\"")


def main():
    print("=== Unifying language prompts for CL-LoRA experiments ===\n")
    for task, prompt in TASK_PROMPTS.items():
        unify_task(task, prompt)
    print("\nDone. All tasks now use fixed single-instruction prompts.")


if __name__ == "__main__":
    main()
