import argparse, gc, json, os
from pathlib import Path
import torch
from shared import (
    load_and_sample_stimuli, split_scratchpad, run_experiment,
    generate_cached_alignments,
    trait_prompt_builder, parse_trait_ratings,
    compute_trait_indices, compute_trait_bias, ALL_TRAITS,
)

def load_model(model_id, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id} in {dtype}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tok

def make_llama_gen(model, tok, max_new=1024):
    def generate(system_prompt, messages, temperature):
        full = [{"role": "system", "content": system_prompt}] + messages
        input_ids = tok.apply_chat_template(
            full, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(model.device)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        raw = tok.decode(out[0, input_len:], skip_special_tokens=True)
        sp, final = split_scratchpad(raw)
        return {"raw_response": raw, "scratchpad": sp, "final_output": final,
                "metadata": {"model": tok.name_or_path, "input_tokens": input_len,
                             "output_tokens": len(out[0]) - input_len}}
    return generate

def trait_parser(text):
    return {f"trait_{t}": v for t, v in parse_trait_ratings(text).items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--pairs_path", default="./dialect-prejudice/data/pairs/habitual.txt")
    p.add_argument("--n_stimuli", type=int, default=40)
    p.add_argument("--n_reps", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output_dir", default="./results_llama")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available(): print("WARNING: No GPU detected.")
    stimuli = load_and_sample_stimuli(args.pairs_path, args.n_stimuli, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    sp = Path(args.output_dir) / "selected_stimuli.json"
    if not sp.exists(): json.dump(stimuli, open(sp, "w"), indent=2)
    model, tok = load_model(args.model_id, args.dtype)
    model_name = f"llama_{args.model_id.split('/')[-1]}"
    gen = make_llama_gen(model, tok)

    cached = None
    if not args.no_cache and not args.test:
        print("Caching alignment responses (4 system prompt variants)...")
        cached = generate_cached_alignments(gen, args.temperature)

    df = run_experiment(
        generate_fn=gen, model_name=model_name,
        experiment="traits", stimuli=stimuli, prompt_builder=trait_prompt_builder,
        response_parser=trait_parser, n_reps=args.n_reps, temperature=args.temperature,
        output_dir=args.output_dir, resume=not args.no_resume, test_mode=args.test,
        cached_alignments=cached,
    )
    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not args.test:
        idx = compute_trait_indices(df)
        idx.to_csv(Path(args.output_dir) / f"{model_name}_traits_indexed.csv", index=False)
        bias = compute_trait_bias(df)
        bias.to_csv(Path(args.output_dir) / f"{model_name}_trait_bias.csv", index=False)
        print(f"\nTrait bias:\n{bias.to_string()}\nDone.")

if __name__ == "__main__": main()
