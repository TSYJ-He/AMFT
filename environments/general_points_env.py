# /amft-project/environments/general_points_env.py
import random
import re
import operator
from typing import List, Tuple, Dict, Any, Optional
class GeneralPointsEnv:
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    def __init__(
            self,
            target_point: int = 24,
            face_card_rule: str = '10',  # or '11_12_13'
            max_steps: int = 5
    ):
        self.target_point = target_point
        self.face_card_rule = face_card_rule
        self.max_steps = max_steps

        if self.face_card_rule not in ['10', '11_12_13']:
            raise ValueError("face_card_rule must be '10' or '11_12_13'")

        self.deck = [f"{rank} of {suit}" for suit in self.SUITS for rank in self.RANKS]
        self.current_cards: List[str] = []
        self.current_card_values: List[int] = []
        self.steps_taken = 0

    def _card_to_value(self, card_rank: str) -> int:
        if card_rank.isdigit():
            return int(card_rank)
        if card_rank == 'A':
            return 1
        if self.face_card_rule == '10':
            return 10
        else:  # '11_12_13'
            return {'J': 11, 'Q': 12, 'K': 13}[card_rank]

    def _safe_eval(self, expression: str) -> Optional[float]:

        try:
            expression = expression.replace('(', ' ( ').replace(')', ' ) ')
            tokens = expression.split()
            allowed_chars = "0123456789.+-*/() "
            if any(c not in allowed_chars for c in expression):
                return None
            return eval(expression, {"__builtins__": {}}, {})
        except (SyntaxError, ZeroDivisionError, TypeError, NameError):
            return None

    def reset(self) -> Dict[str, Any]:
        self.current_cards = random.sample(self.deck, 4)
        ranks = [card.split(' ')[0] for card in self.current_cards]
        self.current_card_values = sorted([self._card_to_value(r) for r in ranks])
        self.steps_taken = 0

        return {
            "cards": self.current_cards,
            "card_values": self.current_card_values
        }
    def verify_formula(self, formula: str) -> Tuple[str, Optional[float]]:
        try:
            numbers_in_formula = sorted([int(n) for n in re.findall(r'\d+', formula)])
        except (ValueError, TypeError):
            return 'SYNTAX_ERROR', None

        if numbers_in_formula != self.current_card_values:
            return 'ILLEGAL_NUMBER', None
        result = self._safe_eval(formula)
        if result is None:
            return 'SYNTAX_ERROR', None
        if abs(result - self.target_point) < 1e-6:  # Use tolerance for float comparison
            return 'CORRECT', result
        else:
            return 'WRONG_RESULT', result
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        status, result = self.verify_formula(action)
        reward = 0.0
        done = False
        if status == 'CORRECT':
            reward = 5.0
            done = True
        elif status == 'WRONG_RESULT':
            reward = -1.0
        elif status == 'ILLEGAL_NUMBER':
            reward = -2.0
        elif status == 'SYNTAX_ERROR':
            reward = -3.0

        if not done and self.steps_taken >= self.max_steps:
            done = True
            if reward == 0.0:
                reward = -1.0

        observation = {
            "cards": self.current_cards,
            "card_values": self.current_card_values
        }

        info = {
            "status": status,
            "result": result,
            "steps_taken": self.steps_taken
        }
        return observation, reward, done, info
