import os
import re
import subprocess

from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from gimkit import Result, from_vllm
from gimkit import guide as g
from log import get_logger
from openai import OpenAI
from pydantic import BaseModel, field_serializer
from tqdm import tqdm


GIT_COMMIT_ID = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    conclusion: bool
    query: str = ""
    response: str = ""
    model_choice: str = ""
    correct_choice: str = ""
    response_len: int = -1
    error_msg: str = ""
    additional_info: dict = {}


class EvalResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    total: int
    evaluates: int
    corrects: int
    errors: int
    accuracy: float
    calibrated_accuracy: float
    avg_response_len: float
    start_time: datetime
    end_time: datetime
    elapsed_minutes: float = 0.0
    args: Namespace
    git_commit_id: str = GIT_COMMIT_ID
    evaled_items: list[EvalItemResult] = []

    @field_serializer("args")
    def serialize_args(self, value: Namespace) -> dict[str, Any]:
        return vars(value)

    def dump(self, filepath: str | None = None):
        if filepath is None:
            dataset = getattr(self.args, "dataset", {})
            dataset_path = dataset.get("path", "unknown_dataset") if isinstance(dataset, dict) else "unknown_dataset"
            model_name = getattr(self.args, "model_name", "unknown_model")
            filename = f"{model_name}_{dataset_path}_{self.start_time.strftime('%y%m%d-%H%M%S')}.json".replace("/", "_")
            filepath = Path(self.args.output_dir) / filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logger.info(f"Saved evaluation results to {filepath}")


class BaseEvaluator:
    def __init__(self, args: Namespace, dataset: Dataset):
        self.start_time = datetime.now()
        self.dataset = dataset
        self.args = args

    @abstractmethod
    def _form_cot_query(self, question: str, choices: list[str]) -> str: ...

    @abstractmethod
    def _model_call(self, query: str) -> Any: ...

    @abstractmethod
    def _parse_response(self, response: Any) -> tuple[str, str, dict]:
        """Extract the response string, model choice, and any additional info from the model response"""
        ...

    def _evaluate_item(self, item: dict) -> EvalItemResult:
        question, choices, correct_choice = (
            item["question"],
            item["choices"],
            item["correct_choice"],
        )
        query = self._form_cot_query(question, choices)
        try:
            raw_response = self._model_call(query)
            response, model_choice, additional_info = self._parse_response(raw_response)
            conclusion = model_choice == correct_choice
            error_msg = ""
        except Exception as e:
            logger.error(e)
            conclusion = False
            response = "ERROR"
            model_choice = "ERROR"
            error_msg = str(e)
            additional_info = {}
        return EvalItemResult(
            conclusion=conclusion,
            query=query,
            response=response,
            model_choice=model_choice,
            correct_choice=correct_choice,
            response_len=len(response),
            error_msg=error_msg,
            additional_info=additional_info,
        )

    def evaluate(self) -> EvalResult:
        logger.info(f"Starting evaluation with config: {self.args}")
        total = len(self.dataset) if self.args.first_n == -1 else min(self.args.first_n, len(self.dataset))

        evaled_items = []
        if self.args.num_proc <= 1:
            for idx in tqdm(range(total), desc=f"Evaluating {self.args.model_name}"):
                result = self._evaluate_item(self.dataset[idx])
                evaled_items.append(result)
        else:
            with ThreadPoolExecutor(max_workers=self.args.num_proc) as executor:
                results = executor.map(self._evaluate_item, (self.dataset[i] for i in range(total)))
                evaled_items = list(tqdm(results, total=total, desc=f"Evaluating {self.args.model_name}"))

        errors = sum(1 for item in evaled_items if item.error_msg)
        corrects = sum(1 for item in evaled_items if item.conclusion)
        evaluates = len(evaled_items)
        accuracy = corrects / evaluates if evaluates > 0 else 0.0
        calibrated_accuracy = corrects / (evaluates - errors) if (evaluates - errors) > 0 else 0.0
        logger.info(f"Final accuracy over {total} examples: {corrects}/{total} = {accuracy:.4f}")
        self.end_time = datetime.now()
        logger.info(f"Evaluation completed at {self.end_time}")
        return EvalResult(
            total=total,
            evaluates=evaluates,
            corrects=corrects,
            errors=errors,
            accuracy=accuracy,
            calibrated_accuracy=calibrated_accuracy,
            avg_response_len=sum(item.response_len for item in evaled_items if item.response_len != -1) / evaluates
            if evaluates > 0
            else 0.0,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_minutes=(self.end_time - self.start_time).total_seconds() / 60.0,
            args=self.args,
            evaled_items=evaled_items,
        )


class GIMEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        openai_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        self.model = from_vllm(openai_client, model_name=args.model_name)

    def _form_cot_query(self, question: str, choices: list[str]) -> str:
        reasoning_guides = [
            str(idx + 1) + ". " + g(desc="One single thinking step") for idx in range(self.args.reason_budget)
        ]
        prompt = f"Answer the question below. You may use reasoning, reflection, trial and error, and other strategies to find the answer.\n\nQuestion: {question}\n\n"
        if self.args.reason_budget > 0:
            prompt += "Let's think step by step:\n" + "\n".join(reasoning_guides) + "\n\n"
        prompt += "Final answer: " + g.select(choices=choices, name="predicted_choice")
        return prompt

    def _model_call(self, query: str) -> Result:
        result = self.model(
            query,
            temperature=self.args.temperature,
            presence_penalty=self.args.presence_penalty,
            seed=self.args.seed,
            max_tokens=self.args.max_tokens,
        )
        return result

    def _parse_response(self, response: Result) -> tuple[str, str, dict]:
        return (
            str(response),
            response.tags["predicted_choice"].content.strip().strip("().,"),
            {tag.name or str(tag.id): tag.content for tag in response.tags},
        )


class CommonEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def _form_cot_query(self, question: str, choices: list[str]) -> str:
        prompt = (
            "Answer the question below. You may use reasoning, reflection, trial and error, and other strategies to find the answer. Remember to end with `The answer is: xxx`.\n\n"
            f"Question: {question}\n\n"
            f"Choose from the following options: {', '.join(choices)}\n\n"
            "Let's think step by step:\n"
        )
        return prompt

    def _model_call(self, query: str) -> str:
        response = self.model.chat.completions.create(
            model=self.args.model_name, messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        response_str = response.strip()
        model_choice = "ERROR"
        additional_info = {f"line_{i + 1}": line for i, line in enumerate(response_str.splitlines())}

        # 1) Try marker-based extraction: e.g. "The answer is: A", "Final answer: (B)", "Answer: C."
        m = re.search(r"(?:the answer is|final answer|answer)[:\s]*\(?([A-Za-z0-9]+)\)?", response_str, re.IGNORECASE)
        if m:
            model_choice = m.group(1).strip().rstrip(".),")
            additional_info["extracted_by"] = "marker"
            return response_str, model_choice, additional_info

        # 2) Scan lines for a short token like "A", "(A)", "A.", "A)" at line start or alone
        for i, line in enumerate(response_str.splitlines()):
            s = line.strip()
            m2 = re.match(r"^\(?([A-Za-z0-9])\)?[\.|\)]?$", s)
            if m2:
                model_choice = m2.group(1)
                additional_info["extracted_by"] = f"line_scan_{i + 1}"
                return response_str, model_choice, additional_info

        # 3) As a last resort, pick the first token of the last line (useful for free-form answers)
        last_line = response_str.splitlines()[-1] if response_str.splitlines() else response_str
        token = last_line.strip().split()[0] if last_line.strip().split() else ""
        if token:
            model_choice = token.strip().strip("() .,")
            additional_info["extracted_by"] = "last_line_first_token"

        return response_str, model_choice, additional_info


def conduct_eval(args: Namespace, ds: Dataset):
    evaluator = GIMEvaluator(args, ds) if args.is_gim else CommonEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
