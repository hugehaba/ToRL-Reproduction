import json, os, re, glob, csv

SAMPLES = "/root/autodl-tmp/output_torl/samples"
OUT     = "/root/autodl-tmp/torl_artifacts"
os.makedirs(OUT, exist_ok=True)

def find_texts(obj, depth=0):
    """递归找出所有可能是模型输出的长字符串"""
    out = []
    if depth > 6: return out
    if isinstance(obj, str):
        if len(obj) > 80: out.append(obj)
    elif isinstance(obj, dict):
        for k in ["response", "responses", "output", "outputs", "completion",
                  "generated", "text", "content", "answer", "solution", "gen"]:
            if k in obj:
                out += find_texts(obj[k], depth+1)
        if not out:
            for v in obj.values(): out += find_texts(v, depth+1)
    elif isinstance(obj, list):
        for v in obj: out += find_texts(v, depth+1)
    return out

def has_code(t):
    return bool(re.search(r"```python|```\s*\n\s*(?:import|from|def|print)|<code>", t))

def has_output(t):
    return bool(re.search(r"```output|<output>|<interpreter>", t))

def n_code(t):
    return len(re.findall(r"```python|<code>", t))

rows = []
for split in ["train", "test"]:
    files = glob.glob(f"{SAMPLES}/{split}/step_*.json")
    if not files: continue
    files.sort(key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    for fp in files:
        step = int(re.search(r"step_(\d+)", fp).group(1))
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"跳过 {fp}: {e}"); continue
        texts = find_texts(data)
        if not texts: continue
        n = len(texts)
        c = sum(has_code(t) for t in texts)
        o = sum(has_output(t) for t in texts)
        calls = sum(n_code(t) for t in texts)
        rows.append({
            "split": split, "step": step, "n_samples": n,
            "code_rate": round(c/n, 4),
            "output_rate": round(o/n, 4),
            "avg_code_blocks": round(calls/n, 3),
            "avg_len_chars": round(sum(len(t) for t in texts)/n, 1),
        })

if not rows:
    print("!! 没解析出内容，打印 step_1.json 结构：")
    d = json.load(open(f"{SAMPLES}/train/step_1.json", encoding="utf-8"))
    print(json.dumps(d, ensure_ascii=False)[:2000])
    raise SystemExit

csv_path = f"{OUT}/tool_usage_by_step.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"{'split':<6}{'step':>6}{'n':>7}{'code_rate':>11}{'out_rate':>10}{'blocks':>8}")
for r in rows:
    if r["split"] == "train" and r["step"] % 20 and r["step"] not in (1,): continue
    print(f"{r['split']:<6}{r['step']:>6}{r['n_samples']:>7}{r['code_rate']:>11.1%}{r['output_rate']:>10.1%}{r['avg_code_blocks']:>8.2f}")

tr = [r for r in rows if r["split"] == "train"]
if tr:
    print(f"\n=== train code_rate: step {tr[0]['step']} = {tr[0]['code_rate']:.1%}  →  step {tr[-1]['step']} = {tr[-1]['code_rate']:.1%}  ({(tr[-1]['code_rate']-tr[0]['code_rate'])*100:+.1f}pp) ===")
te = [r for r in rows if r["split"] == "test"]
if te:
    print(f"=== test  code_rate: step {te[0]['step']} = {te[0]['code_rate']:.1%}  →  step {te[-1]['step']} = {te[-1]['code_rate']:.1%}  ({(te[-1]['code_rate']-te[0]['code_rate'])*100:+.1f}pp) ===")
print(f"\nCSV: {csv_path}")

# 画图
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for split, style in [("train","-"), ("test","o-")]:
        d = [r for r in rows if r["split"]==split]
        if d: ax.plot([r["step"] for r in d], [r["code_rate"]*100 for r in d],
                      style, label=f"{split} code usage", alpha=.8, markersize=4)
    ax.set_xlabel("Training step"); ax.set_ylabel("Code usage rate (%)")
    ax.set_title("ToRL: Tool (code) usage rate over training")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/tool_usage_curve.png", dpi=150)
    print(f"图: {OUT}/tool_usage_curve.png")
except Exception as e:
    print(f"画图跳过: {e}")
