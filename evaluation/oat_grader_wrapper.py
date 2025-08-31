# /amft-project/evaluation/oat_grader_wrapper.py
import logging
import re
from typing import List, Optional, Any
from sympy import sympify, SympifyError
logger = logging.getLogger(__name__)
class OATGraderWrapper:
    def __init__(self, task_name: str):
        self.task_name = task_name
        logger.info(f"Initialized Grader for task: '{self.task_name}'.")

    def grade(self, generated_texts: List[str], ground_truth_texts: List[str]) -> List[float]:
        scores = []
        for gen_text, gt_text in zip(generated_texts, ground_truth_texts):
            try:
                # Extract answers from both the generated text and the ground truth
                gen_answer = self._extract_answer(gen_text)
                gt_answer = self._extract_answer(gt_text)

                if gen_answer is None or gt_answer is None:
                    is_correct = False
                else:
                    is_correct = self._are_answers_equivalent(gen_answer, gt_answer)

                scores.append(1.0 if is_correct else 0.0)
            except Exception as e:
                logger.error(f"Error grading generated text: '{gen_text}'. Error: {e}")
                scores.append(0.0)
        return scores
    def _extract_answer(self, text: str) -> Optional[str]:

        match = re.search(r'\\boxed\{(.+?)\}', text)
        if match:
            return match.group(1).strip()
        numbers = re.findall(r'[-+]?\d+(?:/\d+)?(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].strip()

        match = re.search(r'(?i)(?:the final answer is|the answer is)[:\s]*([^\n]+)', text)
        if match:
            return match.group(1).strip().rstrip('.')

        return None

    def _are_answers_equivalent(self, gen_answer: str, gt_answer: str) -> bool:

        gen_answer = gen_answer.replace(",", "").strip()
        gt_answer = gt_answer.replace(",", "").strip()
        if gen_answer.lower() == gt_answer.lower():
            return True

        if self.task_name in ["math_reasoning", "general_points"]:
            try:
                gen_expr = sympify(gen_answer)
                gt_expr = sympify(gt_answer)
                if abs(gen_expr.evalf() - gt_expr.evalf()) < 1e-6:
                    return True
            except (SympifyError, TypeError, AttributeError):
                pass

        return False
