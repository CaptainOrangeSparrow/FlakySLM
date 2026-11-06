import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (DEFAULT_BENCHMARK_DIR, DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH,
                    PROMPT_STYLES, build_messages, build_scorecard, gold_sub_category,
                    legacy_parse, load_env_file, load_few_shot, load_rows, parse_prediction,
                    prompt_key, read_jsonl)

PROVIDER_DEFAULTS = {
    "anthropic": ("claude-opus-5.5", "claude-opus-5-5", 4096),
    "deepseek": ("deepseek-v4.1-flash", "deepseek-flash", 2048),
    "openai": ("gpt-6-astra", "gpt-6-astra", 4096),
    "dummy": ("dummy", "dummy", 256),
}


class AnthropicClient:
    def __init__(self, model, max_tokens, temperature, effort):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=8)
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        if temperature is not None:
            print("[!] --temperature ignored for Anthropic: claude-opus-5-5 rejects sampling params.")

    def __call__(self, system, messages):
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
            output_config={"effort": self.effort},
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return dict(text=text, input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                    served_model=r.model, stop_reason=r.stop_reason)


class DeepSeekClient:
    def __init__(self, model, max_tokens, temperature, effort):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com", max_retries=8)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def __call__(self, system, messages):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}] + messages)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        r = self.client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        return dict(text=choice.message.content or "", input_tokens=r.usage.prompt_tokens,
                    output_tokens=r.usage.completion_tokens, served_model=r.model,
                    stop_reason=choice.finish_reason,
                    reasoning=getattr(choice.message, "reasoning_content", None))


class OpenAIClient:
    def __init__(self, model, max_tokens, temperature, effort):
        from openai import OpenAI
        self.client = OpenAI(max_retries=8)
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.effort = temperature, effort

    def __call__(self, system, messages):
        kwargs = dict(model=self.model, instructions=system, input=messages,
                      max_output_tokens=self.max_tokens, store=False)
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        r = self.client.responses.create(**kwargs)
        stop = r.status
        if r.status == "incomplete" and r.incomplete_details is not None:
            stop = "max_tokens" if r.incomplete_details.reason == "max_output_tokens" else r.incomplete_details.reason
        if any(getattr(c, "type", "") == "refusal"
               for item in r.output if getattr(item, "type", "") == "message" for c in item.content):
            stop = "refusal"
        return dict(text=r.output_text or "", input_tokens=r.usage.input_tokens,
                    output_tokens=r.usage.output_tokens, served_model=r.model, stop_reason=stop)


class DummyClient:
    def __init__(self, *a, **k):
        self.lock, self.rng = threading.Lock(), random.Random(0)

    def __call__(self, system, messages):
        with self.lock:
            lab = self.rng.choice(["Flaky", "Non-Flaky"])
        cat = "General Flakiness" if lab == "Flaky" else "None"
        return dict(text=f"Classification: {lab}\nRoot Cause Category: {cat}\nReasoning: dummy, non-deterministic.",
                    input_tokens=100, output_tokens=20, served_model="dummy", stop_reason="end_turn")


PROVIDERS = {"anthropic": AnthropicClient, "openai": OpenAIClient, "deepseek": DeepSeekClient,
             "dummy": DummyClient}
API_KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
EFFORT_PROVIDERS = ("anthropic", "openai")


def call_once(client, system, messages):
    t0 = time.time()
    try:
        out = client(system, messages)
        out["error"] = None
    except Exception as e:
        out = dict(text="", input_tokens=0, output_tokens=0, served_model=None,
                   stop_reason=None, error=f"{type(e).__name__}: {e}")
    out["latency_s"] = round(time.time() - t0, 3)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, required=True)
    ap.add_argument("--model", default=None, help="API model id override")
    ap.add_argument("--model-key", default=None, help="short name used in run folder / scorecard")
    ap.add_argument("--data", default=str(DEFAULT_TEST_PATH))
    ap.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    ap.add_argument("--prompt-style", choices=PROMPT_STYLES, default="formatted",
                    help="formatted = dataset prompt + output-format instructions (default); "
                         "dataset = dataset prompt verbatim, as the local pipeline sends it")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples drawn from TRAIN")
    ap.add_argument("--limit", type=int, default=None, help="stratified row subset for pilots")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", default=None,
                    help="DeepSeek default 0; OpenAI/Anthropic default omitted (their reasoning "
                         "models reject it); a float or \"none\"")
    ap.add_argument("--effort", default="low",
                    choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort for Anthropic and OpenAI (default low). Anthropic accepts "
                         "low..max; OpenAI's accepted set depends on the model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    ap.add_argument("--run-tag", default=None, help="default: zero_shot_baseline / <k>_shot_baseline")
    ap.add_argument("--resume", default=None, help="existing run folder to continue")
    return ap.parse_args()


def main():
    args = parse_args()
    load_env_file()
    needed = API_KEY_VARS.get(args.provider)
    if needed and not os.environ.get(needed):
        sys.exit(f"[-] {needed} is not set. Put it in the repo-root .env file (see .env.example) "
                 f"or export it in your shell.")
    if args.provider == "anthropic" and args.effort in ("none", "minimal"):
        sys.exit("[-] Anthropic effort must be one of low, medium, high, xhigh, max.")
    default_key, default_model, default_max = PROVIDER_DEFAULTS[args.provider]
    model = args.model or default_model
    model_key = args.model_key or (default_key if not args.model else model)
    max_tokens = args.max_tokens or default_max
    t = args.temperature if args.temperature is not None else ("0" if args.provider == "deepseek" else "none")
    temperature = None if t.lower() == "none" else float(t)
    run_tag = args.run_tag or ("zero_shot_baseline" if args.shots == 0 else f"{args.shots}_shot_baseline")
    if args.limit:
        run_tag += f"_n{args.limit}"

    if args.resume:
        run_dir = Path(args.resume)
        cfg = json.loads((run_dir / "run_config.json").read_text())
        now = dict(provider=args.provider, model=model, prompt_style=args.prompt_style, shots=args.shots,
                   limit=args.limit, seed=args.seed, data=args.data, max_tokens=max_tokens)
        for k, v in now.items():
            if cfg.get(k) != v:
                sys.exit(f"[-] --resume: '{k}' was {cfg.get(k)!r} in the original run, now {v!r}; pass the same flags.")
        model_key, run_tag = cfg["model_key"], cfg["run_tag"]
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.benchmark_dir) / f"run_{model_key}_{run_tag}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "responses_cache.jsonl"

    rows = load_rows(args.data, limit=args.limit, seed=args.seed)
    few_shot = load_few_shot(args.train, args.shots, seed=args.seed)
    prompts = {}
    row_keys = []
    for rec in rows:
        system, messages = build_messages(rec, args.prompt_style, few_shot)
        k = prompt_key(system, messages)
        prompts.setdefault(k, (system, messages))
        row_keys.append(k)

    cache = {r["prompt_key"]: r for r in read_jsonl(cache_path) if not r.get("error")}
    todo = [k for k in prompts if k not in cache]

    prev_cfg = json.loads((run_dir / "run_config.json").read_text()) if args.resume else {}
    config = dict(
        provider=args.provider, model=model, model_key=model_key, run_tag=run_tag,
        prompt_style=args.prompt_style, shots=args.shots, limit=args.limit, seed=args.seed,
        data=args.data, max_tokens=max_tokens,
        temperature=temperature if args.provider == "deepseek" else None,
        effort=args.effort if args.provider in EFFORT_PROVIDERS else None,
        workers=args.workers, n_rows=len(rows), n_unique_prompts=len(prompts),
        system_prompt=next(iter(prompts.values()))[0],
        started=prev_cfg.get("started", datetime.now().isoformat(timespec="seconds")),
        wall_time_sec=prev_cfg.get("wall_time_sec", 0.0),
    )
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"[*] {run_dir.name}")
    print(f"[*] {len(rows)} rows -> {len(prompts)} unique prompts; {len(cache)} cached, {len(todo)} to call")

    client = PROVIDERS[args.provider](model, max_tokens, temperature, args.effort)
    lock = threading.Lock()

    def work(k):
        system, messages = prompts[k]
        out = call_once(client, system, messages)
        out["prompt_key"] = k
        with lock, cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        return out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(work, k) for k in todo]), 1):
            out = fut.result()
            if out["error"]:
                print(f"  [!] {out['error'][:160]}")
            elif out.get("stop_reason") in ("max_tokens", "length", "refusal"):
                print(f"  [!] stop_reason={out['stop_reason']} (prompt {out['prompt_key'][:8]})")
            else:
                cache[out["prompt_key"]] = out
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")
    config["wall_time_sec"] = round(config["wall_time_sec"] + time.time() - t0, 2)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    last_err = {}
    for r in read_jsonl(cache_path):
        if r.get("error"):
            last_err[r["prompt_key"]] = r
    missing = {k for k in prompts if k not in cache}

    logs = []
    for rec, k in zip(rows, row_keys):
        resp = cache.get(k) or last_err.get(k) or {}
        text = resp.get("text", "")
        pred, pred_cat, method = parse_prediction(text)
        logs.append({
            "sample_index": rec["_sample_index"],
            "project_url": rec.get("project_url", ""),
            "test_identifier": rec.get("test_identifier", ""),
            "ground_truth": rec.get("label", "non_flaky").lower(),
            "predicted_label": pred,
            "sub_category": gold_sub_category(rec),
            "parsing_error": method == "unparsed",
            "prompt_tokens": resp.get("input_tokens", 0),
            "completion_tokens": resp.get("output_tokens", 0),
            "raw_output": text,
            "predicted_category": pred_cat,
            "parse_method": method,
            "predicted_label_legacy_parser": legacy_parse(text),
            "stop_reason": resp.get("stop_reason"),
            "api_error": resp.get("error") if k in missing else None,
            "latency_s": resp.get("latency_s"),
            "served_model": resp.get("served_model"),
            "prompt_key": k,
        })
    with (run_dir / "predictions_log.jsonl").open("w", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r) + "\n")

    n = len(logs)
    p_tok = sum(r["prompt_tokens"] for r in logs)
    c_tok = sum(r["completion_tokens"] for r in logs)
    uniq = [cache[k] for k in prompts if k in cache]
    lat = [u["latency_s"] for u in uniq]
    operational = {
        "total_inference_time_sec": config["wall_time_sec"],
        "avg_latency_per_sample_sec": round(config["wall_time_sec"] / len(prompts), 4) if prompts else 0.0,
        "total_prompt_tokens": p_tok,
        "total_completion_tokens": c_tok,
        "total_tokens_consumed": p_tok + c_tok,
        "avg_tokens_per_sample": round((p_tok + c_tok) / n, 2) if n else 0.0,
        "mean_request_latency_sec": round(sum(lat) / len(lat), 3) if lat else 0.0,
        "concurrent_workers": args.workers,
        "api_calls": len(uniq),
        "api_input_tokens": sum(u["input_tokens"] for u in uniq),
        "api_output_tokens": sum(u["output_tokens"] for u in uniq),
    }
    served = sorted({u.get("served_model") for u in uniq if u.get("served_model")})
    card = build_scorecard(
        logs, model_name=model, model_key=model_key, run_tag=run_tag, operational=operational,
        extra={"provider": args.provider, "served_models": served, "prompt_style": args.prompt_style,
               "shots": args.shots, "api_errors": len(missing),
               "stop_reasons": {s: sum(u.get("stop_reason") == s for u in uniq)
                                for s in sorted({str(u.get("stop_reason")) for u in uniq})}},
    )
    (run_dir / "scorecard_summary.json").write_text(json.dumps(card, indent=2))

    o, d = card["overall_classification_metrics"], card["deduplicated_classification_metrics"]
    print("\n" + "=" * 60)
    print(f" Model          : {model_key} ({', '.join(served) or model})")
    print(f" Saved Results  : {run_dir}")
    print(f" All rows  (n={n:>4}) F1 {o['f1_score']:.4f}  P {o['precision']:.4f}  R {o['recall']:.4f}  Acc {o['accuracy']:.4f}")
    print(f" Unique    (n={d['n']:>4}) F1 {d['f1_score']:.4f}  BalAcc {d['balanced_accuracy']:.4f}  MCC {d['mcc']:.4f}")
    print(f" Parsing Errors : {card['parsing_errors']} / {n}   parse methods: {card['parse_methods']}")
    print(f" API            : {operational['api_calls']} calls, "
          f"{operational['api_input_tokens']:,} in / {operational['api_output_tokens']:,} out tokens, "
          f"{operational['mean_request_latency_sec']}s mean latency")
    if missing:
        print(f" [!] {len(missing)} prompts still failing (scored as non_flaky). Re-run with --resume {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
