import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

# 4-bit NF4 quantization config
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def get_hf_llm(
    model_name: str = "VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    # Load model với quantization 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,  # với một số model LLaMA tokenizer fast có thể lỗi
        padding_side="right",
    )

    # HuggingFace pipeline cho text generation
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    # Wrap bằng LangChain
    llm = HuggingFacePipeline(pipeline=model_pipeline)
    return llm
