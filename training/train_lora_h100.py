#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora_h100.py — bf16 LoRA SFT (DDP 8×H100, 멀티 아키텍처 대응)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EC2에서 검증된 레시피(r16/α32/dropout.05, lr2e-4 cosine w0.05, ep1, seed42)를
H100×8 bf16 DDP로 이식. cohort JSONL(messages 포맷) 그대로 사용.

스모크 (1 GPU, ~3분):
  python train_lora_h100.py --model_name google/gemma-4-31b-it \
      --data <project root>/data/train_neg_1to10_cohort.jsonl \
      --out  <project root>/outputs/smoke --max_steps 10 --max_seq 4096

본학습 (8 GPU):
  torchrun --nproc_per_node 8 train_lora_h100.py \
      --model_name google/gemma-4-31b-it \
      --data <project root>/data/train_neg_1to10_cohort.jsonl \
      --out  <project root>/outputs/gemma4-31b-lora-neg-1to10-ep1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json, argparse, random


def build_text(tok, messages):
    """messages(2 or 3턴) → 학습용 전체 텍스트. 템플릿 실패 시 수동 조립."""
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=False)
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
        return "\n".join(parts) + (tok.eos_token or "")


def guess_lora_targets(model):
    """실제 nn.Linear 인스턴스만 타깃 (래퍼 클래스 대응: gemma-4 ClippableLinear 등).
    - 일반 아키텍처: q_proj 등 leaf가 nn.Linear → 'q_proj'
    - 래퍼 아키텍처: q_proj(래퍼) 내부의 .linear가 nn.Linear → 'q_proj.linear'"""
    import torch.nn as nn
    cand = {"q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv_proj", "out_proj", "w1", "w2", "w3"}
    hits = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        if parts[-1] in cand or (len(parts) >= 2 and parts[-2] in cand):
            hits.append(name)   # 전체 경로명 — 래퍼 오매칭 원천 차단
    lang = [n for n in hits if "language_model" in n]
    if lang:                     # 멀티모달: 언어 모델만 타깃 (비전 타워 제외)
        return lang
    hits = [n for n in hits if not any(t in n for t in
            ("vision", "image", "audio", "mm_projector", "multi_modal"))]
    return hits or ["q_proj", "v_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_seq", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--max_steps", type=int, default=-1, help="스모크용 (-1=전체)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    random.seed(args.seed); torch.manual_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # ── 토크나이저 (멀티모달 계열은 Processor 경유) ──
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception:
        tok = AutoProcessor.from_pretrained(args.model_name,
                                            trust_remote_code=True).tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ── 데이터: messages → text ──
    rows = []
    with open(os.path.expanduser(args.data), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages") or rec.get("train", {}).get("messages")
            if not msgs:
                continue
            rows.append({"text": build_text(tok, msgs)})
    if is_main:
        print(f"[data] {len(rows)} samples | 예시 앞 200자:\n{rows[0]['text'][:200]}")
    ds = Dataset.from_list(rows)

    # ── 모델 (bf16, 아키텍처 자동) ──
    from transformers import AutoModelForImageTextToText
    load_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        load_kw["device_map"] = {"": local_rank}
    model = None; last_err = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        for attn in dict.fromkeys([args.attn, "sdpa", "eager"]):
            try:
                model = cls.from_pretrained(args.model_name,
                                            attn_implementation=attn, **load_kw)
                if is_main:
                    print(f"[load] {cls.__name__} + {attn}")
                break
            except Exception as e:
                last_err = e
        if model is not None:
            break
    if model is None:
        raise last_err
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    targets = guess_lora_targets(model)
    if is_main:
        print(f"[lora] {len(targets)}개 모듈 타깃 (예: {targets[:2]} ...)")
    lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_dropout=args.lora_dropout, bias="none",
                       task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lconf)
    if is_main:
        model.print_trainable_parameters()

    sft = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch" if args.max_steps < 0 else "no",
        report_to=[],
        seed=args.seed,
        max_length=args.max_seq,
        dataset_text_field="text",
        packing=False,
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         processing_class=tok)
    trainer.train()
    if is_main:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"[done] adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
