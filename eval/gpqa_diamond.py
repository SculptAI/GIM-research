from datasets import load_dataset
from utils import get_args, conduct_eval
import random


def _format_gpqa(example: dict, seed: int) -> dict:
    question = example["Question"].strip()
    answers = [
        example["Correct Answer"].strip(),
        example["Incorrect Answer 1"].strip(),
        example["Incorrect Answer 2"].strip(),
        example["Incorrect Answer 3"].strip(),
    ]
    indices = list(range(len(answers)))
    random.seed(seed + hash(question))
    random.shuffle(indices)

    question_with_answer_options = f"{question}\n\nChoices:\n"
    for idx in indices:
        question_with_answer_options += f"({chr(ord('A') + idx)}) {answers[idx]}\n"

    letter_choices = [chr(ord("A") + i) for i in range(len(answers))]
    correct_choice = chr(ord("A") + indices.index(0))

    return {
        "question": question_with_answer_options,
        "choices": letter_choices,
        "correct_choice": correct_choice,
    }


if __name__ == "__main__":
    args = get_args()
    args.dataset = {"path": "Idavidrein/gpqa", "name": "gpqa_diamond", "split": "train"}

    ds = load_dataset(
        args.dataset["path"], args.dataset["name"], split=args.dataset["split"]
    ).map(lambda x: _format_gpqa(x, seed=args.seed))

    conduct_eval(args, ds)
