# prompt-optimizer

**Discover & benchmark LLM prompts via genetic algorithms**

## Installation

```bash
git clone <repo-url>
cd prompt-optimizer
pip install -e .
```

## Quickstart

### Tune a prompt

Run the genetic algorithm to discover a high-performing prompt tail:

```bash
prompt-opt tune \
  --gens 5 \
  --pop 15 \
  --k 10 \
  --sc 2 \
  --penalty 0.0002
```

This will print out the best sequence of instruction snippets.

### Benchmark the best prompt

Evaluate the final prompt on GSM-8K with self-consistency:

```bash
prompt-opt bench \
  --k 50 \
  --sc 5
```

You’ll get:

* **Dev accuracy** on the sampled dev set
* **Total eval time** and per-LLM-call latency
* **Approximate token/word counts**

## How it works

1. **Few‑shot prefix**: You define a `PREFIX` with a short few‑shot (GSM‑8K) intro.
2. **Snippets library**: A curated list of human‑readable instruction snippets (e.g. "Let's think step by step.").
3. **GA search**: Each individual is a sequence of snippet‑indices. The GA uses mutation, crossover, and selection to evolve the best tail by maximizing a fitness score (accuracy minus length penalty).
4. **Evaluation**: `gsm8k_accuracy` runs self‑consistency sampling via an LLM backend (Ollama or OpenAI) to estimate prompt performance.
5. **Benchmarking**: A CLI command to measure real‑world latency, token usage, and accuracy trade‑offs.

## CLI Reference

After installation, the `prompt-opt` command provides two subcommands:

### `tune`

```bash
prompt-opt tune [OPTIONS]
```

* `--gens`: Number of GA generations (default: 10)
* `--pop`: Population size (default: 30)
* `--k`: Dev‑sample size per individual (default: 10)
* `--sc`: Self‑consistency trials (default: 2)
* `--penalty`: Length penalty coefficient (default: 0.002)
* `--backend`: LLM provider (`ollama` or `openai`)
* `--model`: LLM model ID (default: `mistral:7b-instruct`)

### `bench`

```bash
prompt-opt bench [OPTIONS]
```

* `--k`: Dev‑sample size (default: 50)
* `--sc`: Self‑consistency trials (default: 5)

## Tips & FAQs

* **GPU acceleration**: For fastest tuning, use an Ollama GPU model on an RTX 4060 or similar. Pull with `ollama pull mistral:7b-instruct`.
* **Quantization**: Try `mistral:7b-instruct-q4_0` for smaller memory footprint and faster inference.
* **Customize snippets**: Edit `SNIPPETS` in `src/ga.py` to add or remove instruction phrases.
* **Adjust penalty**: Use `--penalty` to balance prompt length vs. accuracy.
