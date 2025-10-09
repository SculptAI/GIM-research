from argparse import ArgumentParser


def _add_model_args(parser):
    parser.add_argument("--model_name", type=str, default="", help="Model under evaluation")
    parser.add_argument("--api_key", type=str, default="", help="API key for the model")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the model API",
    )


def _add_gim_args(parser):
    parser.add_argument("--is_gim", action="store_true", help="Whether to use GIM models")
    parser.add_argument(
        "--reason_budget",
        type=int,
        default=0,
        help="Number of reasoning steps to include in the prompt",
    )


def _add_sample_args(parser):
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=1.0,
        help="Presence penalty for the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens for the model response",
    )


def _add_evaluator_args(parser):
    parser.add_argument("--seed", type=int, default=16, help="Random seed for reproducibility")
    parser.add_argument(
        "--first_n",
        type=int,
        default=-1,
        help="Evaluate only the first n samples. -1 means all",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel evaluation",
    )


def get_args():
    parser = ArgumentParser()
    _add_model_args(parser)
    _add_gim_args(parser)
    _add_sample_args(parser)
    _add_evaluator_args(parser)
    return parser.parse_args()
