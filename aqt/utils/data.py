import random
from argparse import Namespace
from typing import Generator, List

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

NpIntArrayT = NDArray[np.int_]


# FIXME: 校准集处理是否可以与llmcompressor统一
def prepare_calibration_samples(args: Namespace) -> Tensor:
    already_processed = args.quant_data_path.endswith(".pt")
    if already_processed:
        calibration_samples: Tensor = torch.load(args.quant_data_path)
        samples_count = calibration_samples.shape[0]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=False,
        )
        eot_id = tokenizer.eos_token_id

        samples_count = 0
        samples = []

        for x in tokenize_external(
            tokenizer=tokenizer,
            file_path=args.quant_data_path,
            seq_length=args.quant_context_length,
            eot=eot_id,
        ):
            samples_count += 1
            samples.append(x)

        calibration_samples = torch.vstack(samples)

    if not already_processed:
        torch.save(calibration_samples, args.quant_data_save_path)

    idxes = random.sample(range(calibration_samples.shape[0]), args.quant_samples_num)
    calibration_samples = calibration_samples[
        torch.tensor(idxes), : args.quant_context_length
    ]
    calibration_samples_: List[Tensor] = [
        sample.unsqueeze(0) for sample in calibration_samples
    ]

    return calibration_samples_


CHAT_TEMPLATE = "{}"  # TODO: reseved for future


def tokenize_external(
    tokenizer, file_path: str, seq_length: int, eot: int
) -> Generator[torch.Tensor, None, None]:
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for para in tqdm(f.read().split("\n\n\n")):
            content += tokenizer(CHAT_TEMPLATE.format(para)).input_ids + [eot]

    for chunk in chunks(content, seq_length):
        if len(chunk) == seq_length:
            yield torch.tensor(chunk, dtype=torch.int32)


def chunks(lst: List[int], n: int) -> Generator[List[int], None, None]:
    """yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
