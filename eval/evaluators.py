import os

from abc import abstractmethod
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from datasets import Dataset
from log import get_logger
from openai import OpenAI
from pydantic import BaseModel, field_serializer
from tqdm import tqdm

from gimkit import Result, from_vllm
from gimkit import guide as g


logger = get_logger(__name__)


class EvalItemResult(BaseModel):
    conclusion: bool
    query: str = ""
    response: str = ""
    model_choice: str = ""
    correct_choice: str = ""
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
    start_time: datetime
    end_time: datetime
    elapsed_minutes: float = 0.0
    args: Namespace
    evaled_items: list[EvalItemResult] = []

    @field_serializer("args")
    def serialize_args(self, value: Namespace) -> dict[str, Any]:
        return vars(value)

    def dump(self, filepath: str | None = None):
        if filepath is None:
            dataset = getattr(self.args, "dataset", {})
            dataset_path = dataset.get("path", "unknown_dataset") if isinstance(dataset, dict) else "unknown_dataset"
            model_name = getattr(self.args, "model_name", "unknown_model")
            filename = f"{dataset_path}_{model_name}_{self.start_time.strftime('%y%m%d-%H%M%S')}.json".replace("/", "_")
            filepath = "results/" + filename
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
        prompt = f"Answer the question below.\n\nQuestion: {question}\n\n"
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
            response.tags["predicted_choice"].content.strip(),
            {tag.name or str(tag.id): tag.content for tag in response.tags},
        )


class CommonEvaluator(BaseEvaluator):
    def __init__(self, args: Namespace, dataset: Dataset):
        super().__init__(args, dataset)
        self.model = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def _form_cot_query(self, question: str, choices: list[str]) -> str:
        prompt = (
            "Answer the question below. Remember to end with `The answer is: xxx`.\n\n"
            f"Question: {question}\n\n"
            f"Choices: {', '.join(choices)}\n\n"
            "Let's think step by step:\n"
        )
        return prompt

    def _model_call(self, query: str) -> str:
        response = self.model.chat.completions.create(
            model=self.args.model_name,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        response_str = response.strip()
        model_choice = "ERROR"
        if "The answer is:" in response_str:
            model_choice = response_str.split("The answer is:")[-1].strip().split()[0]
        return response_str, model_choice, {f'line_{i+1}': line for i, line in enumerate(response_str.splitlines())}


def conduct_eval(args: Namespace, ds: Dataset):
    evaluator = GIMEvaluator(args, ds) if args.is_gim else CommonEvaluator(args, ds)
    result = evaluator.evaluate()
    result.dump()
