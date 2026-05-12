from pathlib import Path
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
import modelopt.torch.opt as mto

from ..aqt.utils.common import seed_everything
from ..utils.global_args import GlobalConfig

SEED = 42
seed_everything(SEED)
mto.enable_huggingface_checkpointing()

class ModelOptimizerQuantizer():
    def __init__(self, config: GlobalConfig):
        self.model_path = config.raw_config["base_model_path"]
        self.quant_config = config.raw_config['quantization']
        self.data_path = self.quant_config["calib_data_path"]
        self.export_path = Path(config.raw_config["workspace"]["base_dir"]) / config.raw_config["workspace"]["best_weights_dir"]
        self.disabled_layers = config.raw_config["strategy"]["initial_fallback_layers"] + config.raw_config["disable_names"]
        self.device = self._normalize_device(self.quant_config['device'], self.quant_config['visible_devices'])

    
    def run(self):
        """
        执行完整的量化流程。
        """
        QUANT_CFG = {
            "int8": mtq.INT8_DEFAULT_CFG,
            "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
            "fp8": mtq.FP8_DEFAULT_CFG,
            "int4_awq": mtq.INT4_AWQ_CFG,
            "nvfp4": mtq.NVFP4_DEFAULT_CFG,
            "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
            "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        }

        model, tokenizer = self._set_up_model_and_tokenizer(self.model_path)
        data_loader = get_dataset_dataloader(
            dataset_name=self.quant_config["calib_data_path"],
            tokenizer=tokenizer,
            batch_size=self.quant_config["batch_size"],
            num_samples=self.quant_config["calib_samples"],
            device=self.device,
            include_labels=True
        )

        quantization_formats = [QUANT_CFG[q.strip()] for q in self.quant_config["search_space"].split(",")]
        model, _ = mtq.auto_quantize(
            model,
            constraints={"effective_bits": self.quant_config["effective_bits"]},
            data_loader=data_loader,
            forward_step=lambda m, b: m(**b),
            loss_func=lambda out, batch: out.loss,
            quantization_formats=quantization_formats,
            num_calib_steps=len(data_loader),
            num_score_steps=len(data_loader),
            verbose=True,
            disabled_layers=self.disabled_layers
        )
        mtq.print_quant_summary(model)

        # export model
        self._export_model(model, self.export_path)


    def _normalize_device(self, device: str, visible_devices: str) -> str:
        if device == "cpu":
            return "cpu"
        visible_devices = visible_devices.split(",")[0].strip()
        return f"{device}:{visible_devices}"
    
    def _set_up_model_and_tokenizer(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Returns model in eval mode and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self.device,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    def _export_model(self, model: AutoModelForCausalLM, export_path: Path | str) -> None:
        """
        Export model to HuggingFace format.
        """
        if "npu" in self.device:
            model = model.to("cpu")
        
        model.save_pretrained(str(export_path))