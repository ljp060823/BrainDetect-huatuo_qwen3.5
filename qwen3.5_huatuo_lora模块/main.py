from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch

# ================== 配置区 ==================
model_name = "/data/huatuo_qwen3.5_9b/qwen3.5_9b"          # ← 如果你想用 Base 版改成 Qwen/Qwen3.5-9B-Base
dataset_path = "/data/huatuo_qwen3.5_9b/data_propress/train_datasets"       # 上一步保存的路径
output_dir = "/data/huatuo_qwen3.5_9b/qwen3.5_9b_huatuo_qlora"

max_seq_length = 2048
batch_size = 2                          # 根据显存调整（24GB 建议 1-4）
gradient_accumulation = 8
epochs = 2
learning_rate = 2e-4
lora_rank = 32
lora_alpha = 32
# =========================================

# 4bit 配置（QLoRA）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# LoRA 配置（Qwen3.5 专用 target_modules）
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj", "v_proj"]) # ,"o_proj", "gate_proj", "up_proj", "down_proj",

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 加载数据集
dataset = load_from_disk(dataset_path)

# 格式化函数（Qwen 会自动应用 chat_template）
def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return text

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    optim="adamw_8bit", 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #peft_config=lora_config,
    formatting_func=formatting_func,
    args=training_args,
)

print("开始QLoRA微调")
trainer.train()

# 保存 LoRA 权重
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f" 微调完成,LoRA 权重保存在：{output_dir}")