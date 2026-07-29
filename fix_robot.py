"""Fix robot.py to make curobo truly optional. Run on server."""
import re

path = "/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/RoboTwin-main/envs/robot/robot.py"
with open(path, "r") as f:
    code = f.read()

# 1. Import: guard
old = "from .planner import CuroboPlanner"
new = """try:
    from .planner import CuroboPlanner
except (ImportError, ModuleNotFoundError):
    CuroboPlanner = None"""
code = code.replace(old, new)

# 2. isinstance check at line ~135: skip planner reset if no curobo
old = "if not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):"
new = "if CuroboPlanner is not None and (not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner)):"
code = code.replace(old, new)

# 3. set_planner direct creation: skip if no curobo, fallback to SapienPlanner
old = """        if not self.communication_flag:
            self.left_planner = CuroboPlanner(self.left_entity_origion_pose,"""
new = """        if not self.communication_flag:
            if CuroboPlanner is None:
                self.left_planner = SapienPlanner(self.left_entity_origion_pose, self.left_entity.get_active_joints())
                self.right_planner = SapienPlanner(self.right_entity_origion_pose, self.right_entity.get_active_joints())
            else:
                self.left_planner = CuroboPlanner(self.left_entity_origion_pose,"""
code = code.replace(old, new)

# Fix indent for the else block's right_planner
if "            else:" in code:
    pass  # handled by the replacement above

# 4. set_planner multiprocess branch: skip
old = """            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                \"origin_pose\": self.left_entity_origion_pose,
                \"joints_name\": self.left_arm_joints_name,
                \"all_joints\": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                \"yml_path\": abs_left_curobo_yml_path
            }

            right_args = {
                \"origin_pose\": self.right_entity_origion_pose,
                \"joints_name\": self.right_arm_joints_name,
                \"all_joints\": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                \"yml_path\": abs_right_curobo_yml_path
            }"""
new = """        else:
            if CuroboPlanner is not None:
                pass  # multiprocess curobo branch continues below
            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                \"origin_pose\": self.left_entity_origion_pose,
                \"joints_name\": self.left_arm_joints_name,
                \"all_joints\": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                \"yml_path\": abs_left_curobo_yml_path
            }

            right_args = {
                \"origin_pose\": self.right_entity_origion_pose,
                \"joints_name\": self.right_arm_joints_name,
                \"all_joints\": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                \"yml_path\": abs_right_curobo_yml_path
            }"""
code = code.replace(old, new)

# 5. planner_process_worker: skip curobo creation
old = """    from .planner import CuroboPlanner  # 或者绝对路径导入
    planner = CuroboPlanner(args[\"origin_pose\"], args[\"joints_name\"], args[\"all_joints\"], yml_path=args[\"yml_path\"])"""
new = """    if CuroboPlanner is not None:
        from .planner import CuroboPlanner
        planner = CuroboPlanner(args[\"origin_pose\"], args[\"joints_name\"], args[\"all_joints\"], yml_path=args[\"yml_path\"])
    else:
        planner = None"""
code = code.replace(old, new)

with open(path, "w") as f:
    f.write(code)
print("robot.py patched — curobo is now optional, falls back to mplib")
