import argparse, json, os, re, time
from pathlib import Path
import anthropic
import pandas as pd
from shared import (
    load_and_sample_stimuli, split_scratchpad, build_trial_schedule,
    build_system_prompt, ALIGNMENT_QUESTION,
    trait_prompt_builder, parse_trait_ratings,
    compute_trait_indices, compute_trait_bias, ALL_TRAITS,
)

MODEL_ID_DEFAULT = "claude-sonnet-4-6"

def trait_parser(text):
    return {f"trait_{t}": v for t, v in parse_trait_ratings(text).items()}

def submit_batch(client, requests, label):
    print(f"Submitting {label} ({len(requests)} requests)...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    return batch

def poll_batch(client, batch_id, label, poll_interval=30):
    print(f"Polling {label} batch {batch_id}...")
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
    """Strip scratchpad tags from alignment response before feeding into Turn 2."""
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

def run_test(model_id, stimuli):
    client = anthropic.Anthropic()
    trials = build_trial_schedule(stimuli[:1], n_reps=1)
    print(f"{'='*60}\nTEST MODE: {len(trials)} trials (synchronous)\n{'='*60}")
    for trial in trials:
        sys_p = build_system_prompt(trial["oversight"], trial["scratchpad"])
        r1 = client.messages.create(model=model_id, max_tokens=512, temperature=1.0,
            system=sys_p, messages=[{"role": "user", "content": ALIGNMENT_QUESTION}])
        align = r1.content[0].text
        r2 = client.messages.create(model=model_id, max_tokens=2048, temperature=1.0,
            system=sys_p, messages=[
                {"role": "user", "content": ALIGNMENT_QUESTION},
                {"role": "assistant", "content": clean_alignment(align)},
                {"role": "user", "content": trait_prompt_builder(trial)}])
        raw = r2.content[0].text
        sp, final = split_scratchpad(raw)
        parsed = trait_parser(final)
        n_ok = sum(1 for v in parsed.values() if v is not None)
        print(f"\n{'='*60}")
        print(f"TRIAL: {trial['dialect'].upper()} | {trial['oversight']} | {trial['scratchpad']}")
        print(f"Stimulus: \"{trial['stimulus_text']}\"")
        print(f"\n--- ALIGNMENT ---\n{align[:300]}")
        if sp: print(f"\n--- SCRATCHPAD ---\n{sp[:300]}")
        print(f"\n--- TASK OUTPUT ---\n{final[:300]}")
        print(f"\n--- PARSED: {n_ok}/{len(ALL_TRAITS)} traits ---")
    print(f"\n{'='*60}\nTEST COMPLETE\n{'='*60}")

def run_batch(model_id, stimuli, n_reps, temperature, output_dir):
    client = anthropic.Anthropic()
    os.makedirs(output_dir, exist_ok=True)
    mn = f"claude_{model_id}"
    trials = build_trial_schedule(stimuli, n_reps)
    print(f"Model: {mn} | Trials: {len(trials)}")

    # Phase 1: Alignment
    p1_path = Path(output_dir) / f"{mn}_traits_phase1.json"
    if p1_path.exists():
        with open(p1_path) as f: align_results = json.load(f)
        print("Phase 1 loaded from cache.")
    else:
        reqs = [{"custom_id": t["trial_id"]+"_a", "params": {"model": model_id, "max_tokens": 512,
            "temperature": temperature, "system": build_system_prompt(t["oversight"], t["scratchpad"]),
            "messages": [{"role": "user", "content": ALIGNMENT_QUESTION}]}} for t in trials]
        b1 = submit_batch(client, reqs, "Phase 1 (alignment)")
        poll_batch(client, b1.id, "Phase 1")
        align_results = collect_results(client, b1.id)
        json.dump(align_results, open(p1_path, "w"))
        print(f"Phase 1 done: {len(align_results)} responses.")

    # Phase 2: Traits (with cleaned alignment in context)
    p2_path = Path(output_dir) / f"{mn}_traits_phase2.json"
    if p2_path.exists():
        with open(p2_path) as f: task_results = json.load(f)
        print("Phase 2 loaded from cache.")
    else:
        reqs = []
        for t in trials:
            raw_align = align_results.get(t["trial_id"]+"_a", {}).get("text", "Yes, I treat all people equally.")
            reqs.append({"custom_id": t["trial_id"]+"_t", "params": {"model": model_id, "max_tokens": 2048,
                "temperature": temperature, "system": build_system_prompt(t["oversight"], t["scratchpad"]),
                "messages": [{"role": "user", "content": ALIGNMENT_QUESTION},
                    {"role": "assistant", "content": clean_alignment(raw_align)},
                    {"role": "user", "content": trait_prompt_builder(t)}]}})
        b2 = submit_batch(client, reqs, "Phase 2 (traits)")
        poll_batch(client, b2.id, "Phase 2")
        task_results = collect_results(client, b2.id)
        json.dump(task_results, open(p2_path, "w"))
        print(f"Phase 2 done: {len(task_results)} responses.")

    rows = []
    for t in trials:
        ad = align_results.get(t["trial_id"]+"_a", {})
        td = task_results.get(t["trial_id"]+"_t", {})
        raw = td.get("text", ""); sp, final = split_scratchpad(raw)
        parsed = trait_parser(final)
        # Fallback parsing
        if all(v is None for v in parsed.values()):
            parsed = trait_parser(raw)
        rows.append({
            "trial_id": t["trial_id"], "trial_idx": t["trial_idx"],
            "model": mn, "experiment": "traits",
            "stimulus_id": t["stimulus_id"], "dialect": t["dialect"],
            "oversight": t["oversight"], "scratchpad": t["scratchpad"],
            "rep": t["rep"], "stimulus_text": t["stimulus_text"],
            "stimulus_aae": t["stimulus_aae"], "stimulus_sae": t["stimulus_sae"],
            "verb": t["verb"], "source": t["source"],
            "alignment_response": ad.get("text", ""),
            "raw_response": raw, "scratchpad_text": sp, "final_output": final,
            "api_metadata": json.dumps({"align_in": ad.get("input_tokens",0),
                "align_out": ad.get("output_tokens",0), "task_in": td.get("input_tokens",0),
                "task_out": td.get("output_tokens",0)}),
            **parsed})

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{mn}_traits_raw.csv", index=False)
    n_ok = df[[c for c in df.columns if c.startswith("trait_")]].notna().mean().mean()
    print(f"Saved {len(df)} rows. Parse rate: {n_ok:.1%}")
    idx = compute_trait_indices(df)
    idx.to_csv(Path(output_dir) / f"{mn}_traits_indexed.csv", index=False)
    bias = compute_trait_bias(df)
    bias.to_csv(Path(output_dir) / f"{mn}_trait_bias.csv", index=False)
    print(f"\nTrait bias:\n{bias.to_string()}\nDone.")

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
