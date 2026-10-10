"""
resample_groups_1to10_persona_en.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2 cohort data generator — ENGLISH adaptation of
resample_groups_1to10_persona_v2.py (logic preserved).

Input : negative merged data (top-level: domain, polarity,
        persona{...}, survey{questions[...]}, responses[list|dict])
Output: 1-to-10 respondent cohort bundles in messages format
        (user / assistant), English prompts.

Changes vs v2 (Korean):
  - All prompt templates translated to English
    ("Respondent k:", "Q{n}:", format rules)
  - Generic cohort header (no first-persona age/gender claim,
    since cohorts mix demographics; personas carry demographics)
  - persona 'attitude' key supported (fallback: consumption_attitude)

Run:
  python resample_groups_1to10_persona_en.py \
      --input  ~/ACL_EN/neg_train_final.jsonl \
      --output ~/ACL_EN/train_neg_1to10_cohort.jsonl \
      --target_size 5000 --min_size 1 --max_size 10 --seed 42
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List


# ─────────────────────────────────────────────────────────────
# Persona → prompt text (name removed to avoid identity leakage)
# ─────────────────────────────────────────────────────────────
def persona_to_text(persona: Dict) -> str:
    lines = []
    p = persona

    polarity = p.get("polarity", "")
    if polarity:
        lines.append(f"Polarity: {polarity}")

    age = p.get("age_group", "")
    gender = p.get("gender", "")
    if age or gender:
        demo = []
        if age: demo.append(f"Age: {age}")
        if gender: demo.append(f"Gender: {gender}")
        lines.append(" / ".join(demo))

    income = p.get("income_level", "")
    household = p.get("household", "")
    if income or household:
        eco = []
        if income: eco.append(f"Income: {income}")
        if household: eco.append(f"Household: {household}")
        lines.append(" / ".join(eco))

    occupation = p.get("occupation", "")
    region = p.get("region", "")
    if occupation or region:
        social = []
        if occupation: social.append(f"Occupation: {occupation}")
        if region: social.append(f"Region: {region}")
        lines.append(" / ".join(social))

    lifestyle = p.get("lifestyle_tags", [])
    if lifestyle:
        lines.append(f"Lifestyle: {', '.join(lifestyle)}")

    bg = p.get("background", "")
    if bg:
        lines.append(f"Background: {bg}")

    # EN data uses 'attitude'; keep KR-era fallback for safety
    attitude = p.get("attitude", "") or p.get("consumption_attitude", "")
    if attitude:
        lines.append(f"Attitude: {attitude}")

    pain = p.get("pain_points", [])
    if pain:
        lines.append(f"Pain points: {', '.join(pain)}")

    return "\n  ".join(lines)


# ─────────────────────────────────────────────────────────────
# Response value normalization (keep "number. text" as-is)
# ─────────────────────────────────────────────────────────────
def normalize_response_value(resp: str) -> str:
    return str(resp).strip()


# ─────────────────────────────────────────────────────────────
# Survey questions → prompt text
# ─────────────────────────────────────────────────────────────
def format_questions_text(questions: List[Dict]) -> str:
    out_lines = []
    for q in questions:
        q_idx = q.get('questionOrder')
        q_text = q.get('questionText', '')
        q_type = q.get('questionType') or q.get('questionFormatType', '')

        line = f"Q{q_idx}: {q_text}"
        opts = q.get('options', []) or q.get('selectOptions', [])

        if q_type == "OPEN_ENDED":
            line += "\n(Answer freely.)"
        elif opts:
            is_multi = q_type in ("CATEGORICAL_MULTI", "RANKED_MULTI")
            if is_multi:
                line += "\n(Multiple selections allowed)"

            opt_lines = []
            for o in opts:
                opt_lines.append(f"{o['optionOrder']}. {o['optionText']}")
            line += "\n" + "\n".join(opt_lines)

            if is_multi:
                line += "\n(Separate multiple selections with commas.)"
            else:
                line += "\n(Select exactly one option.)"

        out_lines.append(line)

    return "\n\n".join(out_lines)


# ─────────────────────────────────────────────────────────────
# One respondent's responses → text (dict AND list supported)
# ─────────────────────────────────────────────────────────────
def responses_to_cohort_format(responses) -> str:
    if isinstance(responses, dict):
        sorted_keys = sorted(responses.keys(), key=lambda x: int(x))
        lines = []
        for qid in sorted_keys:
            val = normalize_response_value(responses[qid])
            lines.append(f"Q{qid}: {val}")
        return "\n".join(lines)

    if isinstance(responses, list):
        sorted_items = sorted(
            responses,
            key=lambda x: int(x.get("questionOrder", 0)) if isinstance(x, dict) else 0,
        )
        lines = []
        for item in sorted_items:
            if not isinstance(item, dict):
                continue
            qid = item.get("questionOrder", "?")
            val_str = (item.get("response_text") or item.get("selected_text")
                       or str(item.get("selected_code", "")))
            val = normalize_response_value(val_str)
            lines.append(f"Q{qid}: {val}")
        return "\n".join(lines)

    return ""


def build_cohort_assistant(responses_list: List[Dict]) -> str:
    blocks = []
    for k, resp in enumerate(responses_list, 1):
        text = responses_to_cohort_format(resp)
        blocks.append(f"Respondent {k}:\n{text}")
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────
# User prompt (cohort context + per-respondent personas)
# ─────────────────────────────────────────────────────────────
def build_cohort_user_message(
    cohort_size: int,
    questions_text: str,
    personas: List[Dict],
) -> str:
    context = (
        f"You are {cohort_size} independent survey respondents in the United States.\n"
        f"Each respondent (Respondent 1 to Respondent {cohort_size}) is a separate "
        f"individual with their own background, preferences, and values.\n"
        f"Each respondent's persona is described below. Answer in a way that is "
        f"consistent with each persona.\n\n"
    )

    persona_blocks = []
    for k, p in enumerate(personas, 1):
        text = persona_to_text(p)
        block = f"[Respondent {k}]\n  {text}"
        persona_blocks.append(block)
    persona_section = "\n\n".join(persona_blocks) + "\n\n"

    intro = "Please answer the following survey.\n\n"

    format_rule = f"""

Write the answers for ALL questions in the following format:

Respondent 1:
Q1: [number. option text, or a free-text answer]
Q2: [number. option text, or a free-text answer]
...

Respondent 2:
Q1: ...
...

Multiple choice: answer strictly as "number. option text" (e.g., "2. Dissatisfied", "3. Neutral"). For multi-select questions, separate items with commas (e.g., "1. A, 3. B").
Open-ended: answer with a word or a short sentence.
Do not add any text outside the specified format.
"""

    return context + persona_section + intro + questions_text + format_rule


# ─────────────────────────────────────────────────────────────
# Cohort creation (same logic as v2)
# ─────────────────────────────────────────────────────────────
def create_random_groups(
    samples: List[Dict],
    target_count: int = 5000,
    min_size: int = 1,
    max_size: int = 10,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)

    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    print(f"[1/3] grouping {len(samples)} samples by domain...")
    skipped = 0
    for s in samples:
        resp = s.get("responses")
        questions = s.get("survey", {}).get("questions")
        if not questions:
            skipped += 1
            continue
        if isinstance(resp, dict) and not resp:
            skipped += 1
            continue
        if not resp:
            has_resp_in_q = any(
                isinstance(q, dict) and (q.get("response_text") or q.get("selected_text"))
                for q in questions
            )
            if not has_resp_in_q:
                skipped += 1
                continue
            s["responses"] = questions
        domain = s.get("domain", "unknown")
        by_domain[domain].append(s)

    print(f"  domains: {len(by_domain)}")
    print(f"  skipped: {skipped} samples (no responses/questions)")
    for d, samps in sorted(by_domain.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"    {d[:50]}: {len(samps)} samples")

    polarity_counts = Counter()
    for d, samps in by_domain.items():
        for s in samps:
            pol = s.get("polarity") or s.get("persona", {}).get("polarity", "other")
            polarity_counts[pol] += 1
    print(f"  polarity distribution: {dict(polarity_counts)}")

    print(f"[2/3] creating {target_count} cohorts (size {min_size}-{max_size}, uniform)...")
    domain_names = list(by_domain.keys())
    domain_weights = [len(by_domain[d]) for d in domain_names]

    groups: List[Dict] = []
    attempts = 0
    while len(groups) < target_count and attempts < target_count * 5:
        attempts += 1
        domain = rng.choices(domain_names, weights=domain_weights, k=1)[0]
        pool = by_domain[domain]
        if len(pool) < min_size:
            continue

        size = rng.randint(min_size, min(max_size, len(pool)))
        selected = rng.sample(pool, size)

        first = selected[0]
        respondent_personas = [s.get("persona", {}) for s in selected]
        respondent_responses = [s.get("responses", {}) for s in selected]

        groups.append({
            "domain": domain,
            "questions": first["survey"]["questions"],
            "personas": respondent_personas,
            "responses": respondent_responses,
            "cohort_size": size,
            "polarities": [
                p.get("polarity", "other") for p in respondent_personas
            ],
        })

    print(f"  cohorts created: {len(groups)}")

    size_dist = Counter(g["cohort_size"] for g in groups)
    print(f"  cohort size distribution:")
    for s in sorted(size_dist):
        print(f"    size {s}: {size_dist[s]}")

    return groups


# ─────────────────────────────────────────────────────────────
def group_to_training_sample(group: Dict) -> Dict:
    questions_text = format_questions_text(group["questions"])
    user_text = build_cohort_user_message(
        cohort_size=group["cohort_size"],
        questions_text=questions_text,
        personas=group["personas"],
    )
    assistant_text = build_cohort_assistant(group["responses"])

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "_meta": {
            "domain": group["domain"],
            "cohort_size": group["cohort_size"],
            "polarities": group["polarities"],
        },
    }


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_size", type=int, default=5000)
    ap.add_argument("--min_size", type=int, default=1)
    ap.add_argument("--max_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.input}")
    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  loaded: {len(samples)} samples")

    groups = create_random_groups(
        samples, target_count=args.target_size,
        min_size=args.min_size, max_size=args.max_size, seed=args.seed,
    )

    print(f"[3/3] writing training samples: {args.output}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for g in groups:
            sample = group_to_training_sample(g)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  saved: {len(groups)} samples")

    print(f"\n[stats]")
    print(f"  input samples : {len(samples)}")
    print(f"  output cohorts: {len(groups)}")
    print(f"  avg cohort size: {sum(g['cohort_size'] for g in groups)/len(groups):.2f}")
    print(f"  total respondent exposures: {sum(g['cohort_size'] for g in groups)}")

    print(f"\n[first sample preview]")
    first = group_to_training_sample(groups[0])
    meta = first["_meta"]
    print(f"  cohort_size: {meta['cohort_size']}")
    print(f"  domain: {meta['domain'][:60]}")
    print(f"  polarities: {meta['polarities']}")
    print(f"\n  -- USER (first 1500 chars) --")
    print(first['messages'][0]['content'][:1500])
    print(f"\n  -- USER (last 300 chars) --")
    print(first['messages'][0]['content'][-300:])
    print(f"\n  -- ASSISTANT (first 600 chars) --")
    print(first['messages'][1]['content'][:600])


if __name__ == "__main__":
    main()
