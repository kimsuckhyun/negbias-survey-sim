import glob, json, os
from collections import Counter
c = Counter(); dropped = 0
for f in sorted(glob.glob(os.path.expanduser("~/ACL_EN/output/train_data_*.jsonl"))):
    good = []
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip(): continue
            try:
                s = json.loads(line); good.append(line); c[s.get("polarity","?")] += 1
            except Exception:
                dropped += 1
    tmp = f + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        out.write("\n".join(good) + ("\n" if good else ""))
    os.replace(tmp, f)
neg = c["strongly_negative"] + c["weakly_negative"]
print(f"dropped(broken): {dropped}")
print(f"strongly_negative: {c['strongly_negative']}")
print(f"weakly_negative  : {c['weakly_negative']}")
print(f"other            : {sum(c.values()) - neg}")
print(f"NEG TOTAL        : {neg}")
