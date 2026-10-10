#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multi_score.py — 임의 모델(base 또는 +adapter)의 OpinionQA 94문항 logprob 판독
출력: dist CSV (q_id, option_num, p) — 기존 eval_nrr_gemma4.py metrics와 호환

사용:
  python eval_multi_score.py --model_name google/gemma-4-31b-it \
      --questions eval/converted/eval_questions.json --out eval/dists/gemma4-31b_base.csv
  python eval_multi_score.py --model_name google/gemma-4-31b-it \
      --adapter outputs/gemma4-31b-neg-1to10-ep1 \
      --questions eval/converted/eval_questions.json --out eval/dists/gemma4-31b_ft.csv
"""
import os, json, argparse


def build_prompt(q):
    opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(q["options"]))
    return (
        "You are 1 independent survey respondents in the United States.\n"
        "Answer as a typical adult respondent.\n\n"
        "Please answer the following survey.\n\n"
        f"Q1: {q['question']}\n{opts}\n(Select exactly one option.)\n\n"
        "Write the answers in the following format:\n"
        "Respondent 1:\nQ1: [number. option text]\n"
        'Multiple choice: answer strictly as "number. option text".\n'
        "Do not add any text outside the specified format.\n"
    )


def digit_token_ids(tok, k):
    """옵션 번호 1..k 각각의 첫 토큰 id (선행 공백 변형 포함 후보 중 단일 토큰만)."""
    ids = []
    for d in range(1, k + 1):
        cands = []
        for s in (str(d), " " + str(d)):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                cands.append(enc[0])
        ids.append(cands)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoProcessor,
                              AutoModelForCausalLM, AutoModelForImageTextToText)

    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception:
        tok = AutoProcessor.from_pretrained(args.model_name,
                                            trust_remote_code=True).tokenizer

    load_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
                   device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = None; last = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        for attn in dict.fromkeys([args.attn, "sdpa", "eager"]):
            try:
                model = cls.from_pretrained(args.model_name,
                                            attn_implementation=attn, **load_kw)
                print(f"[load] {cls.__name__} + {attn}")
                break
            except Exception as e:
                last = e
        if model is not None:
            break
    if model is None:
        raise last
    if args.adapter and args.adapter.lower() != "none":
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, os.path.expanduser(args.adapter))
            print("[adapter] attached")
        except TypeError:
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file
            import glob as _g
            _ad = os.path.expanduser(args.adapter)
            model = get_peft_model(model, LoraConfig.from_pretrained(_ad))
            _sd = load_file(sorted(_g.glob(os.path.join(_ad, "adapter_model*.safetensors")))[0])
            _m, _u = model.load_state_dict(_sd, strict=False)
            print(f"[adapter] 수동 주입 폴백: lora 텐서 {sum(1 for k in _sd if 'lora' in k)} | unexpected {len(_u)}")
    model.eval()

    questions = json.load(open(os.path.expanduser(args.questions)))
    rows = []
    with torch.no_grad():
        for qi, q in enumerate(questions):
            user = build_prompt(q)
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                text = user + "\n"
            text += "Respondent 1:\nQ1: "
            enc = tok(text, return_tensors="pt").to(model.device)
            logits = model(**enc).logits[0, -1]

            K = len(q["options"])
            dids = digit_token_ids(tok, K)
            vals = []
            for cands in dids:
                if not cands:
                    vals.append(float("-inf")); continue
                vals.append(max(float(logits[c]) for c in cands))
            t = torch.tensor(vals)
            p = torch.softmax(t, dim=0).tolist()
            for oi in range(K):
                rows.append({"q_id": q["q_id"], "option_num": oi + 1,
                             "p": round(p[oi], 6)})
            if (qi + 1) % 20 == 0:
                print(f"  {qi+1}/{len(questions)}", flush=True)

    import pandas as pd
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.expanduser(args.out), index=False)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
