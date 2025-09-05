# /amft-project/environments/virl_env.py
import random
from typing import List, Tuple, Dict, Any, Optional
class VirlEnv:
    def __init__(
            self,
            action_space_mode: str = 'absolute',  # 'absolute' or 'relative'
            max_steps: int = 10
    ):
        if action_space_mode not in ['absolute', 'relative']:
            raise ValueError("action_space_mode must be 'absolute' or 'relative'")

        self.action_space_mode = action_space_mode
        self.max_steps = max_steps

        self.route = {
            'instructions': [
                "Turn to face east.",
                "Move forward until you see 'Cafe Bene'.",
                "Turn to face north.",
                "Move forward to your destination."
            ],
            'path': [
                {'pos': (0, 0), 'landmark': None, 'expert_action': 'turn_direction(east)'},
                {'pos': (1, 0), 'landmark': 'Cafe Bene', 'expert_action': 'turn_direction(north)'},
                {'pos': (1, 1), 'landmark': 'Destination', 'expert_action': 'stop()'}
            ],
            'start_pos': (0, 0),
            'start_orientation': 'north'
        }
        # ----------------------------

        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_orientation: str = 'north'
        self.steps_taken: int = 0
        self.current_instruction_index: int = 0
        self.action_history: List[str] = []

    def reset(self) -> Dict[str, Any]:
        self.agent_pos = self.route['start_pos']
        self.agent_orientation = self.route['start_orientation']
        self.steps_taken = 0
        self.current_instruction_index = 0
        self.action_history = []

        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        current_path_node = None
        for node in self.route['path']:
            if node['pos'] == self.agent_pos:
                current_path_node = node
                break

        visual_obs = "You are at an intersection."
        if current_path_node and current_path_node['landmark']:
            visual_obs += f" You see '{current_path_node['landmark']}'."

        return {
            "instructions": self.route['instructions'],
            "current_instruction": self.route['instructions'][self.current_instruction_index],
            "visual_observation": visual_obs,
            "action_history": self.action_history.copy()
        }

    def _process_action(self, action: str) -> bool:
        current_node_index = -1
        for i, node in enumerate(self.route['path']):
            if node['pos'] == self.agent_pos:
                current_node_index = i
                break

        if current_node_index == -1:  # Agent is off-path
            return False

        expert_action = self.route['path'][current_node_index]['expert_action']

        if action == expert_action:
            if action.startswith("turn_direction"):
                self.agent_orientation = action[action.find("(") + 1:action.find(")")]
            elif action == "forward()":
                if current_node_index + 1 < len(self.route['path']):
                    self.agent_pos = self.route['path'][current_node_index + 1]['pos']
            if self.current_instruction_index < len(self.route['instructions']) - 1:
                self.current_instruction_index += 1
            return True
        else:
            return False

    def step(self, action: str, landmark_recognized: bool = True) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        """
        self.steps_taken += 1
        self.action_history.append(action)

        reward = 0.0
        done = False
        if not landmark_recognized:
            reward = -1.5

        is_correct_action = False
        if reward == 0.0:
            is_correct_action = self._process_action(action)
            if is_correct_action:
                reward = 1.0
            else:

        if action == "stop()" and self.agent_pos == self.route['path'][-1]['pos']:
            done = True
            reward = 5.0
        elif self.steps_taken >= self.max_steps:
            done = True
            if reward == 0.0:
                reward = -1.0

        new_observation = self._get_observation()
        info = {
            "status": "Success" if is_correct_action else "Failure",
            "steps_taken": self.steps_taken,
            "agent_pos": self.agent_pos,
        }

        return new_observation, reward, done, info
