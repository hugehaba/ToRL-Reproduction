import json, os, re, glob, csv
from collections import Counter

SAMPLES = "/root/autodl-tmp/output_torl/samples"
OUT = "/root/autodl-tmp/torl_artifacts"
os.makedirs(OUT, exist_ok=True)

def load(fp):
    try:
        return json.load(open(fp, encoding="utf-8"))
    except json.JSONDecodeError:
        objs, dec, s, i = [], json.JSONDecoder(), open(fp, encoding="utf-8").read(), 0
        while i < len(s):
            try:
                o, j = dec.raw_decode(s, i); objs.append(o); i = j
                while i < len(s) and s[i] in " \n\r\t": i += 1
            except: break
        return objs

def find_texts(o, d=0):
    r = []
    if d > 6: return r
    if isinstance(o, str):
        if len(o) > 80: r.append(o)
    elif isinstance(o, dict):
        for k in ["response","responses","output","outputs","completion","generated","text","content","answer","solution","gen"]:
            if k in o: r += find_texts(o[k], d+1)
        if not r:
            for v in o.values(): r += find_texts(v, d+1)
    elif isinstance(o, list):
        for v in o: r += find_texts(v, d+1)
    return r

def outputs_of(t):
    return re.findall(r"```output\s*\n(.*?)```", t, re.DOTALL) + \
           re.findall(r"<output>\s*(.*?)</output>", t, re.DOTALL)

def classify(o):
    s = o.strip()
    if not s: return "empty"
    low = s.lower()
    if "unknownerror" in low: return "UnknownError"
    if "timeout" in low or "timed out" in low: return "Timeout"
    if "traceback" in low: return "Traceback"
    if re.search(r"\w*Error\b|\w*Exception\b", s): return "OtherError"
    return "OK"

rows, gerr = [], Counter()
for split in ["train", "test"]:
    fs = glob.glob(f"{SAMPLES}/{split}/step_*.json")
    if not fs: continue
    fs.sort(key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    for fp in fs:
        step = int(re.search(r"step_(\d+)", fp).group(1))
        ts = find_texts(load(fp))
        if not ts: continue
        cnt = Counter()
        for t in ts:
            for o in outputs_of(t):
                c = classify(o); cnt[c] += 1; gerr[c] += 1
        tot = sum(cnt.values())
        if not tot: continue
        rows.append({"split": split, "step": step, "n_exec": tot,
            "ok_rate": round(cnt["OK"]/tot, 4),
            "err_rate": round(1 - cnt["OK"]/tot, 4),
            "unknown": round(cnt["UnknownError"]/tot, 4),
            "timeout": round(cnt["Timeout"]/tot, 4),
            "traceback": round(cnt["Traceback"]/tot, 4)})

with open(f"{OUT}/sandbox_by_step.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("=== 沙盒执行结果总分布 ===")
G = sum(gerr.values())
for k, v in gerr.most_common():
    print(f"  {k:<14} {v:>7}  {v/G:>6.1%}")
print(f"\n{'split':<6}{'step':>6}{'n_exec':>8}{'OK':>8}{'ERR':>8}{'Unknown':>9}{'Timeout':>9}")
for r in rows:
    if r["split"] == "train" and r["step"] % 40 and r["step"] != 1: continue
    print(f"{r['split']:<6}{r['step']:>6}{r['n_exec']:>8}{r['ok_rate']:>8.1%}{r['err_rate']:>8.1%}{r['unknown']:>9.1%}{r['timeout']:>9.1%}")

te = [r for r in rows if r["split"] == "test"]
if te:
    print(f"\n=== test 沙盒成功率: step {te[0]['step']} = {te[0]['ok_rate']:.1%} → step {te[-1]['step']} = {te[-1]['ok_rate']:.1%} ===")

# 相关性：沙盒成功率 vs code usage
try:
    import statistics as st
    tu = {}
    for line in open(f"{OUT}/tool_usage_by_step.csv"):
        p = line.strip().split(",")
        if p[0] == "test" and p[1].isdigit(): tu[int(p[1])] = float(p[3])
    pair = [(r["ok_rate"], tu[r["step"]]) for r in te if r["step"] in tu]
    if len(pair) > 3:
        x, y = zip(*pair)
        r_ = st.correlation(x, y)
        print(f"=== 沙盒成功率 vs code usage 相关系数 r = {r_:.3f}  (n={len(pair)}) ===")
except Exception as e:
    print(f"相关性跳过: {e}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = te
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot([r["step"] for r in d], [r["ok_rate"]*100 for r in d], "o-", label="Sandbox success rate", ms=4)
    if tu:
        s = [r["step"] for r in d if r["step"] in tu]
        ax.plot(s, [tu[k]*100 for k in s], "s--", label="Code usage rate", ms=4, alpha=.8)
    ax.set_xlabel("Training step"); ax.set_ylabel("%"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Sandbox reliability vs. code usage (test)")
    plt.tight_layout(); plt.savefig(f"{OUT}/sandbox_vs_code.png", dpi=150)
    print(f"图: {OUT}/sandbox_vs_code.png")
except Exception as e:
    print(f"画图跳过: {e}")

print(f"\nCSV: {OUT}/sandbox_by_step.csv")
