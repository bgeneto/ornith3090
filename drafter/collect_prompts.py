"""Collect a diverse prompt set for Ornith-1.5-9B (EN technical chat, code: Python/TS/Shell/Docker,
Portuguese instructions, tool calls, and math reasoning) for self-distillation data generation.
Output: data/prompts.jsonl with {"id","src","messages","think"}."""
import json, random, os, glob, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

R = random.Random(1234)
OUT = os.path.join(HERE, "data", "prompts.jsonl")
CACHE = os.path.join(HERE, "data", "hf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def dl(repo, patterns):
    try:
        return snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                                 local_dir=f"{CACHE}/{repo.replace('/', '__')}")
    except Exception as e:
        print(f"Warning: could not download {repo}: {e}")
        return None


def parquet_rows(root, cols=None):
    if not root or not os.path.exists(root):
        return []
    out = []
    for f in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        try:
            out.extend(pq.read_table(f, columns=cols).to_pylist())
        except Exception as e:
            print("skip", f, e)
    return out


prompts = []


def add(src, msgs):
    prompts.append({"src": src, "messages": msgs})


# 1) UltraChat 200k (Technical EN chat)
d = dl("HuggingFaceH4/ultrachat_200k", ["data/train_sft-00000-of-*.parquet"])
uc = parquet_rows(d, ["messages"])
R.shuffle(uc)
n = 0
for r in uc:
    m = r.get("messages", [])
    if not m or m[0]["role"] != "user" or len(m[0]["content"]) < 20:
        continue
    if len(m) >= 3 and R.random() < 0.25:
        add("ultrachat", [{"role": x["role"], "content": x["content"]} for x in m[:3]])
    else:
        add("ultrachat", [{"role": "user", "content": m[0]["content"]}])
    n += 1
    if n >= 2000:
        break
print("ultrachat", n)

# 2) Magicoder OSS-Instruct (Python, TypeScript, Shell, Docker code)
d = dl("ise-uiuc/Magicoder-OSS-Instruct-75K", ["*.parquet", "data/*.parquet", "*.jsonl", "data/*.jsonl"])
mc = parquet_rows(d, ["problem", "lang"])
if not mc and d:
    for f in glob.glob(f"{d}/**/*.jsonl", recursive=True):
        try:
            mc.extend(json.loads(l) for l in open(f))
        except Exception:
            pass
R.shuffle(mc)
for r in mc[:2000]:
    prob = r.get("problem") or r.get("instruction") or ""
    if prob:
        add("code", [{"role": "user", "content": prob}])
print("code", min(2000, len(mc)))

# 3) Portuguese instructions (PT-BR)
d = dl("recogna-nlp/benc-instruct", ["*.parquet", "data/*.parquet"]) or dl("maritaca-ai/mpt-instruct", ["*.parquet"])
pt_rows = parquet_rows(d)
R.shuffle(pt_rows)
n = 0
for r in pt_rows:
    q = r.get("instruction") or r.get("question") or r.get("prompt") or ""
    if q and len(q) > 20:
        add("portuguese", [{"role": "user", "content": q}])
        n += 1
        if n >= 1500:
            break
print("portuguese", n)

# 4) Agent & Tool-calling examples
tool_prompts = [
    "Execute a bash command to find all .ts and .tsx files modified in the last 24 hours.",
    "Write a Dockerfile for a multi-stage production Node.js TypeScript application with alpine base.",
    "Call the search tool to find the documentation for asyncio.TaskGroup in Python 3.11.",
    "Parse the following JSON log file using jq in bash and output top error codes.",
    "Implement a fast binary search algorithm in TypeScript with strict null checks.",
    "Escreva uma função em Python para validar CPF e CNPJ com testes unitários.",
    "Crie um script em shell para verificar o uso de memória e disco de containers Docker.",
    "Write a GitHub Actions workflow to build and push Docker images to GHCR on release tag.",
    "Configure a reverse proxy in Nginx with SSL termination and WebSocket support.",
    "Use the file_search tool to find definition of class AttentionBackend in the repo.",
]
for p in tool_prompts * 50:  # amplify agent/tool vocabulary
    add("agent_tools", [{"role": "user", "content": p}])
print("agent_tools", len(tool_prompts) * 50)

# 5) GSM8K train (Math reasoning)
gs = parquet_rows(os.path.join(REPO, "bench", "quality-data", "gsm8k"), ["question"])
if not gs:
    d = dl("openai/gsm8k", ["main/train-*.parquet"])
    gs = parquet_rows(d, ["question"])
R.shuffle(gs)
for r in gs[:800]:
    q = r.get("question")
    if q:
        add("gsm8k", [{"role": "user", "content": q}])
print("gsm8k", min(800, len(gs)))

R.shuffle(prompts)
with open(OUT, "w") as f:
    for i, p in enumerate(prompts):
        p["id"] = i
        base = 0.7 if p["src"] in ("gsm8k", "code") else 0.4
        p["think"] = R.random() < base
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Total prompts collected: {len(prompts)} ({sum(p['think'] for p in prompts)} with thinking enabled) -> {OUT}")
