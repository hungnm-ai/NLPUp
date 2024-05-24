from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Model name of path to pre-trained model"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name"
        }
    )
    padding_side: Optional[str] = field(
        default="left",
        metadata={
            "help": "Padding side is left or right"
        }
    )
    model_max_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "Model max length"
        }
    )

    lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Where do you want LORA to training model"
        }
    )
    qlora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Where do you want QLoRA to training model"
        }
    )
    flash_attention: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Where do you want Flash Attention to training model"
        }
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path

        if self.padding_side == "right" and self.flash_attention is True:
            raise ValueError("When using flash_attention, padding_side must be 'left'")

        if self.qlora is True:
            self.lora = True


# Load and split dataset
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

dataset = dataset.shuffle()
dataset = dataset["train"].train_test_split(test_size=0.1)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


def print_trainable_parameters(_model: PreTrainedModel):
    """
    Print the number of trainable parameters of model
    """

    trainable_params = 0
    all_params = 0
    for _, param in _model.named_parameters():
        all_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || "
        f"all params: {all_params} || "
        f"trainable%: {round(100 * trainable_params / all_params, 2)}")


def load_tokenizer(model_args: ModelArguments):
    _tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                               padding_side=model_args.padding_side,
                                               use_fast=True,
                                               legacy=True,
                                               model_max_length=model_args.model_max_length,
                                               truncation=True,
                                               truncation_side="left")
    _tokenizer.pad_token = _tokenizer.eos_token


def load_model(
        model_args: ModelArguments,
        tokenizer: PreTrainedTokenizer
) -> PreTrainedModel:
    """
    Load and configure model. Support fine-tuning full weights or LORA, QLoRA
    Args:
        model_args:
        tokenizer:

    Returns:

    """
    device = {"": 0}

    # Check device support bloat16
    compute_type = torch.float16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_type = torch.bfloat16

    # check device support FlashAttention
    attn_implementation = None
    if model_args.flash_attention:
        attn_implementation = "eager"

        major_version, minor_version = torch.cuda.get_device_capability()
        if major_version >= 8:
            attn_implementation = 'flash_attention_2'
            print("Your device supports Flash Attention 2")

    quantization_config = None
    if model_args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_type
        )

    # Load pre-trained model
    _model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        device_map=device,
        torch_dtype=compute_type,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )

    # resize token embedding if adding special tokens
    _model.resize_token_embeddings(len(tokenizer))

    _model.config.pad_token_id = tokenizer.pad_token_id

    # enable gradient checkpointing to save memory
    _model.gradient_checkpointing_enable = True

    if model_args.qlora:
        _model = prepare_model_for_kbit_training(_model)

    # LoRA config based on Sebastian Raschka experiment
    if model_args.lora:
        peft_config = LoraConfig(
            r=256,
            lora_alpha=128,
            target_modules="all_linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        _model = get_peft_model(model=_model, peft_config=peft_config)

    """Because KV cache is useless during training(Finetune), It only works for inference.
        For a Generative Language model.
        For a training iteration, all result are computed parallel with casual mask and teacher-forcing,
        which means all the key and value for different input token are computed in one time.
        https://stackoverflow.com/questions/76633335/why-does-hugging-face-falcon-model-use-mode-config-use-cache-false-why-wouldn
        """
    _model.config.use_cache = False

    print_trainable_parameters(_model)
    return _model
