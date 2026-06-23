# # import os
# # import re
# # import json
# # import argparse
# # import torch
# # import wandb
# #
# # from datasets import load_dataset
# # from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# # from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# # from trl import SFTTrainer
# #
# #
# # # ==========================================
# # # 1. 自动提取特殊 Token (SID 密码)
# # # ==========================================
# # def extract_special_tokens(data_path):
# #     """扫描训练集，找出所有的 SID 特殊符号，如 <a_15>, <x0_10>"""
# #     print(f"正在扫描数据集提取特殊 Token: {data_path}")
# #     with open(data_path, 'r', encoding='utf-8') as f:
# #         data = json.load(f)
# #
# #     text_content = json.dumps(data)
# #     # 正则匹配形如 <a_15> 或 <x0_10> 的符号
# #     special_tokens = list(set(re.findall(r'<[a-z0-9]+_\d+>', text_content)))
# #     print(f"共发现 {len(special_tokens)} 个唯一的 SID Token！")
# #     return special_tokens
# #
# #
# # # ==========================================
# # # 2. 格式化 Prompt (支持 LLaMA-3 模板 / 通用模板)
# # # ==========================================
# # def format_llama_instruction(batch):
# #     """
# #     使用极其清晰的 Chat 模板，防止大模型在特殊控制符上产生幻觉
# #     """
# #     out = []
# #     for ins, inp, resp in zip(batch["instruction"], batch["input"], batch["output"]):
# #         # 采用严谨的 System -> User -> Assistant 结构
# #         text = (
# #             f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
# #             f"{ins.strip()}<|eot_id|>"
# #             f"<|start_header_id|>user<|end_header_id|>\n\n"
# #             f"{inp.strip()}<|eot_id|>"
# #             f"<|start_header_id|>assistant<|end_header_id|>\n\n"
# #             f"{resp.strip()}<|eot_id|>"
# #         )
# #         out.append(text)
# #     return out
# #
# #
# # # ==========================================
# # # 3. 核心训练主循环
# # # ==========================================
# # def train(args):
# #     print("[Step 1] 初始化与路径核对...")
# #     os.environ["WANDB_PROJECT"] = "SA-SID-Alignment"
# #
# #     # 动态挂载新 Token
# #     new_tokens = extract_special_tokens(args.train_data_path)
# #
# #     print(f"[Step 2] 正在加载基座模型与 Tokenizer: {args.model_path}")
# #     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# #
# #     # 词表扩容核心逻辑
# #     num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
# #     print(f"Tokenizer 词表成功扩容，新增 {num_added_toks} 个 SID Tokens。")
# #
# #     # 确保 pad_token 存在，避免报错
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.padding_side = "right"  # 训练时通常使用 right padding
# #
# #     # 加载模型 (使用 bf16 以节省显存并加速)
# #     model = AutoModelForCausalLM.from_pretrained(
# #         args.model_path,
# #         device_map="auto",
# #         torch_dtype=torch.bfloat16,
# #     )
# #
# #     # 极其致命的一步：调整模型 Embedding 大小，匹配刚才扩容的词表！
# #     model.resize_token_embeddings(len(tokenizer))
# #
# #     # 启用梯度检查点 (省显存神器)
# #     model.gradient_checkpointing_enable()
# #     model = prepare_model_for_kbit_training(model)
# #
# #     print("[Step 3] 配置 LoRA 适配器 (注入 embed_tokens)...")
# #     peft_config = LoraConfig(
# #         r=64,
# #         lora_alpha=128,
# #         lora_dropout=0.05,
# #         target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
# #         bias="none",
# #         task_type="CAUSAL_LM",
# #         # 必加参数：因为加入了新词，必须放开底层词向量表和输出头的冷冻状态！
# #         modules_to_save=["embed_tokens", "lm_head"]
# #     )
# #     model = get_peft_model(model, peft_config)
# #     model.print_trainable_parameters()
# #
# #     print("[Step 4] 加载对齐数据集...")
# #     train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
# #     valid_dataset = load_dataset("json", data_files=args.valid_data_path)["train"]
# #     print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")
# #
# #     print("[Step 5] 初始化 Trainer 并开始炼丹...")
# #     training_args = TrainingArguments(
# #         output_dir=args.output_dir,
# #         per_device_train_batch_size=args.batch_size,
# #         gradient_accumulation_steps=args.grad_accum,
# #         num_train_epochs=args.epochs,
# #         learning_rate=args.lr,
# #         #evaluation_strategy="steps",
# #         eval_strategy="steps",
# #         eval_steps=100,
# #         save_steps=200,
# #         logging_steps=10,
# #         warmup_steps=100,
# #         bf16=True,  # 确保显卡支持 bfloat16 (如 A100/RTX3090)，否则改为 fp16=True
# #         run_name=f"SA-SID-Align-{args.datafold}",
# #         report_to="wandb",
# #         remove_unused_columns=False,
# #     )
# #
# #     trainer = SFTTrainer(
# #         model=model,
# #         train_dataset=train_dataset,
# #         eval_dataset=valid_dataset,
# #         args=training_args,
# #         tokenizer=tokenizer,
# #         #processing_class=tokenizer,
# #         formatting_func=format_llama_instruction,  # 使用我们刚才定义的规范模板
# #         max_seq_length=args.cutoff_len,  # 放宽到 2048
# #     )
# #
# #     # 训练前预览一条数据
# #     example = format_llama_instruction(train_dataset[:1])
# #     print("\n[效果预览] 喂给模型的第一条数据长这样：")
# #     print(example[0] + "\n")
# #
# #     # 开炮！
# #     trainer.train()
# #
# #     print(f"[Step 6] 训练完成！正在保存 LoRA 权重至: {args.output_dir}")
# #     trainer.save_model(args.output_dir)
# #     tokenizer.save_pretrained(args.output_dir)
# #     print("Alignment 对齐任务大功告成！")
# #
# #
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     # 核心路径配置
# #     parser.add_argument("--datafold", type=str, default="NOLA", help="Dataset name")
# #     parser.add_argument("--model_path", type=str, default="/home/ljp/work/xhl/QueryExpansion/models/Llama-3-8B-Instruct", help="Path to Base LLM (e.g. LLaMA-3)")
# #
# #     # 我们第一步生成的静态语义对齐数据集
# #     parser.add_argument("--train_data_path", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/train_align.json")
# #     parser.add_argument("--valid_data_path", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/valid_align.json")
# #
# #     # 权重输出目录
# #     parser.add_argument("--output_dir", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/models/align_lora")
# #
# #     # 训练超参数
# #     parser.add_argument("--batch_size", type=int, default=4)
# #     parser.add_argument("--grad_accum", type=int, default=4)
# #     parser.add_argument("--epochs", type=int, default=3)
# #     parser.add_argument("--lr", type=float, default=2e-4)
# #
# #     parser.add_argument("--cutoff_len", type=int, default=2048)  # 扩大上下文窗口
# #
# #     args = parser.parse_args()
# #
# #     train(args)
# import os
# import re
# import json
# import argparse
# import torch
# import wandb
#
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# #最新版核心导入：必须使用 SFTConfig
# from trl import SFTTrainer, SFTConfig
# from transformers import BitsAndBytesConfig
#
# # ==========================================
# # 1. 自动提取特殊 Token (SID 密码)
# # ==========================================
# def extract_special_tokens(data_path):
#     """扫描训练集，找出所有的 SID 特殊符号，如 <a_15>, <x0_10>"""
#     print(f"正在扫描数据集提取特殊 Token: {data_path}")
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     text_content = json.dumps(data)
#     special_tokens = list(set(re.findall(r'<[a-z0-9]+_\d+>', text_content)))
#     print(f"共发现 {len(special_tokens)} 个唯一的 SID Token！")
#     return special_tokens
#
#
# # ==========================================
# # 2. 格式化 Prompt (支持单条与批处理的动态判定)
# # ==========================================
# def format_llama_instruction(example):
#     """
#     使用极其清晰的 Chat 模板，防止大模型在特殊控制符上产生幻觉。
#     具备动态类型判定，完美解决 TRL 内部 map 时传递单条数据的 list/string 类型冲突。
#     """
#     # 场景 A: 如果传入的是一个批次 (Batched - Lists)
#     if isinstance(example["instruction"], list):
#         out = []
#         for ins, inp, resp in zip(example["instruction"], example["input"], example["output"]):
#             text = (
#                 f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#                 f"{ins.strip()}<|eot_id|>"
#                 f"<|start_header_id|>user<|end_header_id|>\n\n"
#                 f"{inp.strip()}<|eot_id|>"
#                 f"<|start_header_id|>assistant<|end_header_id|>\n\n"
#                 f"{resp.strip()}<|eot_id|>"
#             )
#             out.append(text)
#         return out
#
#     # 场景 B: 如果传入的是单条数据 (Single - String)
#     else:
#         ins = example["instruction"]
#         inp = example["input"]
#         resp = example["output"]
#         text = (
#             f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#             f"{ins.strip()}<|eot_id|>"
#             f"<|start_header_id|>user<|end_header_id|>\n\n"
#             f"{inp.strip()}<|eot_id|>"
#             f"<|start_header_id|>assistant<|end_header_id|>\n\n"
#             f"{resp.strip()}<|eot_id|>"
#         )
#         return text
#
#
# # ==========================================
# # 3. 核心训练主循环
# # ==========================================
# def train(args):
#     print("[Step 1] 初始化与路径核对...")
#     os.environ["WANDB_PROJECT"] = "SA-SID-Alignment"
#
#     # 动态挂载新 Token
#     new_tokens = extract_special_tokens(args.train_data_path)
#
#     print(f"[Step 2] 正在加载基座模型与 Tokenizer: {args.model_path}")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#
#     # 词表扩容核心逻辑
#     num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
#     print(f" Tokenizer 词表成功扩容，新增 {num_added_toks} 个 SID Tokens。")
#
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#
#     # 加载模型 (使用 bf16)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#     )
#
#     # 极其致命的一步：调整模型 Embedding 大小
#     model.resize_token_embeddings(len(tokenizer))
#
#     # 启用梯度检查点以节省显存
#     model.gradient_checkpointing_enable()
#     # 兼容性处理，为后续潜在的 LoRA 层做准备
#     #model = prepare_model_for_kbit_training(model)
#     model.gradient_checkpointing_enable()
#     model.config.use_cache = False
#
#
#     print("[Step 3] 配置 LoRA 适配器 (注入 embed_tokens)...")
#     peft_config = LoraConfig(
#         r=64,
#         lora_alpha=128,
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
#         bias="none",
#         task_type="CAUSAL_LM",
#         # 必加参数：因为加入了新词，必须放开底层词向量表和输出头！
#         modules_to_save=["embed_tokens", "lm_head"]
#     )
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#
#     print("[Step 4] 加载对齐数据集...")
#     train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
#     valid_dataset = load_dataset("json", data_files=args.valid_data_path)["train"]
#     print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")
#
#     print("[Step 5] 初始化 Trainer 并开始炼丹...")
#
#     # 最新版规范：所有控制流参数全部进入 SFTConfig
#     training_args = SFTConfig(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         eval_strategy="steps",  # 严禁使用 evaluation_strategy
#         eval_steps=100,
#         save_steps=200,
#         logging_steps=10,
#         warmup_steps=100,
#         bf16=True,
#         run_name=f"SA-SID-Align-{args.datafold}",
#         report_to="none",
#         remove_unused_columns=False,
#         max_length=args.cutoff_len,  # 最新版：长度控制必须写在这里
#     )
#
#     # 最新版规范：Trainer 只接收纯净的数据、模型和统一处理类
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         args=training_args,
#         processing_class=tokenizer,  # 最新版：废除 tokenizer 参数，统一用 processing_class
#         formatting_func=format_llama_instruction,
#     )
#
#     example = format_llama_instruction(train_dataset[:1])
#     print("\n [效果预览] 喂给模型的第一条数据长这样：")
#     print(example[0] + "\n")
#
#     print(" 炼丹炉正式点火！")
#     trainer.train()
#
#     print(f"[Step 6] 训练完成！正在保存 LoRA 权重至: {args.output_dir}")
#     trainer.save_model(args.output_dir)
#     # processing_class 保存，防错兼容
#     if hasattr(trainer, "processing_class"):
#         trainer.processing_class.save_pretrained(args.output_dir)
#     else:
#         tokenizer.save_pretrained(args.output_dir)
#
#     print("Alignment 对齐任务大功告成！")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datafold", type=str, default="NOLA", help="Dataset name")
#
#     # 路径配置
#     parser.add_argument("--model_path", type=str,
#                         default="/home/ljp/work/cyw/models/qwen-7b-insruct", help="Path to Base LLM")
#     parser.add_argument("--train_data_path", type=str,
#                         default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/train_align.json")
#     parser.add_argument("--valid_data_path", type=str,
#                         default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/valid_align.json")
#     parser.add_argument("--output_dir", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/align_lora")
#
#     # # 训练超参数
#     # parser.add_argument("--batch_size", type=int, default=4)
#     # parser.add_argument("--grad_accum", type=int, default=4)
#     # parser.add_argument("--epochs", type=int, default=3)
#     # parser.add_argument("--lr", type=float, default=2e-4)
#     # parser.add_argument("--cutoff_len", type=int, default=2048)
#
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--grad_accum", type=int, default=4)
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--lr", type=float, default=2e-4)
#     parser.add_argument("--cutoff_len", type=int, default=300)
#
#     args = parser.parse_args()
#     train(args)

import os
import re
import json
import argparse
import torch
import wandb

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# 最新版核心导入：必须使用 SFTConfig
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig


# ==========================================
# 1. 自动提取特殊 Token (SID 密码)
# ==========================================
def extract_special_tokens(data_path):
    """扫描训练集，找出所有的 SID 特殊符号，如 <a_15>, <x0_10>"""
    print(f"正在扫描数据集提取特殊 Token: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_content = json.dumps(data)
    special_tokens = list(set(re.findall(r'<[a-z0-9]+_\d+>', text_content)))
    print(f"共发现 {len(special_tokens)} 个唯一的 SID Token！")
    return special_tokens


# ==========================================
# 2. 格式化 Prompt (针对 Qwen 适配原生 ChatML 模板)
# ==========================================
def format_qwen_instruction(example):
    """
    核心修改：Qwen 系列必须使用 ChatML 模板 (<|im_start|> 等)，严禁使用 LLaMA-3 模板！
    """
    if isinstance(example["instruction"], list):
        out = []
        for ins, inp, resp in zip(example["instruction"], example["input"], example["output"]):
            text = (
                f"<|im_start|>system\n{ins.strip()}<|im_end|>\n"
                f"<|im_start|>user\n{inp.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{resp.strip()}<|im_end|>"
            )
            out.append(text)
        return out
    else:
        ins = example["instruction"]
        inp = example["input"]
        resp = example["output"]
        text = (
            f"<|im_start|>system\n{ins.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{inp.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{resp.strip()}<|im_end|>"
        )
        return text


# ==========================================
# 3. 核心训练主循环
# ==========================================
def train(args):
    print("[Step 1] 初始化与路径核对...")
    os.environ["WANDB_PROJECT"] = "SA-SID-Alignment"

    # 动态挂载新 Token
    new_tokens = extract_special_tokens(args.train_data_path)

    print(f" [Step 2] 正在加载基座模型与 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 词表扩容核心逻辑
    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    print(f" Tokenizer 词表成功扩容，新增 {num_added_toks} 个 SID Tokens。")

    # Qwen 的 pad_token 通常自带，如果没有则设定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #  核心修复 1：真正配置并挂载 QLoRA 4-bit 量化（彻底解决 OOM）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 开启 4-bit 加载
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算过程保持 bfloat16 精度
        bnb_4bit_use_double_quant=True,  # 开启双重量化压榨显存
        bnb_4bit_quant_type="nf4",  # 使用 nf4 数据类型
    )

    print(" 正在以 4-bit 模式加载模型，这将大幅节约显存...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=quantization_config,  # 必须传入这个！
    )

    # 调整模型 Embedding 大小以接纳新字
    model.resize_token_embeddings(len(tokenizer))

    #  核心修复 2：只有在 4-bit 模式下，才必须调用 prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("️ [Step 3] 配置 LoRA 适配器 (注入 embed_tokens)...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(" [Step 4] 加载对齐数据集...")
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    valid_dataset = load_dataset("json", data_files=args.valid_data_path)["train"]
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")

    print("⚙️ [Step 5] 初始化 Trainer 并开始炼丹...")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=10,
        warmup_steps=100,
        bf16=True,
        run_name=f"SA-SID-Align-{args.datafold}",
        report_to="none",
        remove_unused_columns=False,
        max_seq_length=args.cutoff_len,  #  核心修复 3：参数必须叫 max_seq_length
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=format_qwen_instruction,  # 使用 Qwen 专属格式
    )

    example = format_qwen_instruction(train_dataset[:1])
    print("\n [效果预览] 喂给模型的第一条数据长这样：")
    print(example[0] + "\n")

    print(" 炼丹炉正式点火！")
    trainer.train()

    print(f" [Step 6] 训练完成！正在保存 LoRA 权重至: {args.output_dir}")
    trainer.save_model(args.output_dir)
    if hasattr(trainer, "processing_class"):
        trainer.processing_class.save_pretrained(args.output_dir)
    else:
        tokenizer.save_pretrained(args.output_dir)

    print(" Alignment 对齐任务大功告成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafold", type=str, default="NOLA", help="Dataset name")

    parser.add_argument("--model_path", type=str,
                        default="/home/ljp/work/cyw/models/qwen-7b-insruct", help="Path to Base LLM")
    parser.add_argument("--train_data_path", type=str,
                        default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/train_align.json")
    parser.add_argument("--valid_data_path", type=str,
                        default="/home/mysjz/mywork/V2-SID/data/NOLA/alignment/valid_align.json")
    parser.add_argument("--output_dir", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/align_lora")

    # 你的防 OOM 超参数设定 (如果引入了 4-bit，其实你可以放心把 cutoff_len 提回到 1024 甚至 2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cutoff_len", type=int, default=5000)  # 建议最低 512，300 可能会切掉太多属性

    args = parser.parse_args()
    train(args)