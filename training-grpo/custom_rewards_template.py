"""Template for custom reward functions in MaxText GRPO post-training.

Save this file to your PVC mount (e.g. /data/custom_rewards.py) or Cloud Storage bucket,
and pass its path to MaxText via:
  - reward_functions_path=/data/custom_rewards.py
  - reward_functions=format_reward,exact_match_reward
"""

from typing import Any, Sequence
import re


def format_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    tmvp_config: Any = None,
    **kwargs,
) -> list[float]:
  """Rewards completion structure: <reasoning>...</reasoning><answer>...</answer>

  Args:
      prompts: List of prompt strings.
      completions: List of model-generated completion strings.
      tmvp_config: Active MaxText configuration object (optional).
      **kwargs: Extra metadata columns passed from the dataset.

  Returns:
      List of float rewards (one per completion).
  """
  rewards = []
  pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

  for completion in completions:
    match = re.search(pattern, completion, re.DOTALL)
    rewards.append(1.0 if match else 0.0)

  return rewards


def exact_match_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    tmvp_config: Any = None,
    **kwargs,
) -> list[float]:
  """Verifies the extracted answer against the dataset's ground truth answer.

  Args:
      prompts: List of prompt strings.
      completions: List of model-generated completion strings.
      tmvp_config: Active MaxText configuration object.
      **kwargs: Extra metadata columns passed from the dataset.
                kwargs.get("answer") contains ground-truth answers if present.

  Returns:
      List of float rewards (one per completion).
  """
  rewards = []
  answers = kwargs.get("answer", [])

  for i, completion in enumerate(completions):
    score = 0.0
    if "<answer>" in completion and "</answer>" in completion:
      extracted = completion.split("<answer>")[-1].split("</answer>")[0].strip()

      if i < len(answers) and answers[i] is not None:
        gold_answer = str(answers[i]).strip()
        if extracted.lower() == gold_answer.lower():
          score = 2.0
      else:
        if len(extracted) > 0:
          score = 1.0

    rewards.append(score)

  return rewards
