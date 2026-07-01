"""
guilt_claude.py — Guilt attribution via Anthropic Batch API.
Usage:
    !python guilt_claude.py --n_reps 5 --output_dir ./results_claude
    !python guilt_claude.py --test
"""
import argparse, json, os, re, time
from pathlib import Path
import anthropic
import pandas as pd
from shared import (
    load_and_sample_stimuli, split_scratchpad, build_trial_schedule,
    build_system_prompt, ALIGNMENT_QUESTION,
    guilt_prompt_builder, parse_guilt_verdict, compute_guilt_bias,
)

MODEL_ID_DEFAULT = "claude-sonnet-4-6"

def submit_batch(client, requests, label):
    print(f"Submitting {label} ({len(requests)} requests)...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    return batch

def poll_batch(client, batch_id, label, poll_interval=30):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        done = c.succeeded + c.errored + c.expired + c.canceled
        total = done + c.processing
        print(f"  [{label}] {done}/{total} ({c.succeeded} ok, {c.errored} err)")
        if batch.processing_status == "ended": return batch
        time.sleep(poll_interval)

def collect_results(client, batch_id):
    results = {}
    for e in client.messages.batches.results(batch_id):
        if e.result.type == "succeeded":
            results[e.custom_id] = {"text": e.result.message.content[0].text,
                "input_tokens": e.result.message.usage.input_tokens,
                "output_tokens": e.result.message.usage.output_tokens}
        else:
            results[e.custom_id] = {"text": f"ERROR: {e.result.type}", "input_tokens": 0, "output_tokens": 0}
    return results

def clean_alignment(text):
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

def run_test(model_id, stimuli):
    client = anthropic.Anthropic()
    trials = build_trial_schedule(stimuli[:1], n_reps=1)
    print(f"{'='*60}\nTEST MODE: {len(trials)} trials\n{'='*60}")
    for t in trials:
        sys_p = build_system_prompt(t["oversight"], t["scratchpad"])
        r1 = client.messages.create(model=model_id, max_tokens=512, temperature=1.0,
            system=sys_p, messages=[{"role": "user", "content": ALIGNMENT_QUESTION}])
        align = r1.content[0].text
        r2 = client.messages.create(model=model_id, max_tokens=1024, temperature=1.0,
            system=sys_p, messages=[
                {"role": "user", "content": ALIGNMENT_QUESTION},
                {"role": "assistant", "content": clean_alignment(align)},
                {"role": "user", "content": guilt_prompt_builder(t)}])
        raw = r2.content[0].text
        sp, final = split_scratchpad(raw)
        parsed = parse_guilt_verdict(final)
        if all(v is None for v in parsed.values()):
            parsed = parse_guilt_verdict(raw)
        print(f"\n{'='*60}\n{t['dialect'].upper()} | {t['oversight']} | {t['scratchpad']}")
        print(f"Stimulus: \"{t['stimulus_text']}\"")
        print(f"Alignment: {align[:200]}")
        if sp: print(f"Scratchpad: {sp[:200]}")
        print(f"Output: {final[:200]}")
        print(f"Parsed: {parsed}")
    print(f"\n{'='*60}\nTEST COMPLETE\n{'='*60}")

def run_batch(model_id, stimuli, n_reps, temperature, output_dir):
    client = anthropic.Anthropic()
    os.makedirs(output_dir, exist_ok=True)
    mn = f"claude_{model_id}"
    trials = build_trial_schedule(stimuli, n_reps)
    print(f"Model: {mn} | Trials: {len(trials)}")

    p1_path = Path(output_dir) / f"{mn}_guilt_phase1.json"
    if p1_path.exists():
        with open(p1_path) as f: align_results = json.load(f)
        print("Phase 1 loaded from cache.")
    else:
        reqs = [{"custom_id": t["trial_id"]+"_a", "params": {"model": model_id, "max_tokens": 512,
            "temperature": temperature, "system": build_system_prompt(t["oversight"], t["scratchpad"]),
            "messages": [{"role": "user", "content": ALIGNMENT_QUESTION}]}} for t in trials]
        b1 = submit_batch(client, reqs, "Phase 1")
        poll_batch(client, b1.id, "Phase 1")
        align_results = collect_results(client, b1.id)
        json.dump(align_results, open(p1_path, "w"))

    p2_path = Path(output_dir) / f"{mn}_guilt_phase2.json"
    if p2_path.exists():
        with open(p2_path) as f: task_results = json.load(f)
        print("Phase 2 loaded from cache.")
    else:
        reqs = []
        for t in trials:
            raw_align = align_results.get(t["trial_id"]+"_a", {}).get("text", "Yes, I treat all people equally.")
            reqs.append({"custom_id": t["trial_id"]+"_t", "params": {"model": model_id, "max_tokens": 1024,
                "temperature": temperature, "system": build_system_prompt(t["oversight"], t["scratchpad"]),
                "messages": [{"role": "user", "content": ALIGNMENT_QUESTION},
                    {"role": "assistant", "content": clean_alignment(raw_align)},
                    {"role": "user", "content": guilt_prompt_builder(t)}]}})
        b2 = submit_batch(client, reqs, "Phase 2")
        poll_batch(client, b2.id, "Phase 2")
        task_results = collect_results(client, b2.id)
        json.dump(task_results, open(p2_path, "w"))

    rows = []
    for t in trials:
        ad = align_results.get(t["trial_id"]+"_a", {})
        td = task_results.get(t["trial_id"]+"_t", {})
        raw = td.get("text", ""); sp, final = split_scratchpad(raw)
        parsed = parse_guilt_verdict(final)
        if all(v is None for v in parsed.values()):
            parsed = parse_guilt_verdict(raw)
        if all(v is None for v in parsed.values()) and sp:
            parsed = parse_guilt_verdict(sp)
        rows.append({**{k: t[k] for k in ["trial_id","trial_idx","stimulus_id","dialect",
            "oversight","scratchpad","rep","stimulus_text","stimulus_aae","stimulus_sae","verb","source"]},
            "model": mn, "experiment": "guilt", "alignment_response": ad.get("text",""),
            "raw_response": raw, "scratchpad_text": sp, "final_output": final,
            **parsed})

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{mn}_guilt_raw.csv", index=False)
    null_rate = df['guilty'].isna().mean()
    print(f"Saved {len(df)} rows. Guilty rate: {df['guilty'].mean():.2%} | Parse failures: {null_rate:.1%}")
    bias = compute_guilt_bias(df)
    bias.to_csv(Path(output_dir) / f"{mn}_guilt_bias.csv", index=False)
    print(f"\nGuilt bias:\n{bias.to_string()}\nDone.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    p.add_argument("--pairs_path", default="./dialect-prejudice/data/pairs/habitual.txt")
    p.add_argument("--n_stimuli", type=int, default=40)
    p.add_argument("--n_reps", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output_dir", default="./results_claude")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY"
    stimuli = load_and_sample_stimuli(args.pairs_path, args.n_stimuli, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    sp = Path(args.output_dir) / "selected_stimuli.json"
    if not sp.exists(): json.dump(stimuli, open(sp, "w"), indent=2)
    if args.test: run_test(args.model_id, stimuli)
    else: run_batch(args.model_id, stimuli, args.n_reps, args.temperature, args.output_dir)

if __name__ == "__main__": main()
