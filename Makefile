PYTHON ?= python3
THREADS ?= 8

.PHONY: test check tokenizer train eval leak param-count generate birth-test experiments decontaminate registry serve rust-test benchmark memory-eval memory-qa rust-mem

benchmark: ## run the ACTIVE core-basic gate on the registered name checkpoint
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/benchmark.py \
	  --checkpoint checkpoints/name/resumed/final.npz --config configs/toy_name.json \
	  --out benchmarks/results

memory-eval: ## retrieval-quality eval for the memory engine (ADVISORY core-retrieval)
	$(PYTHON) tools/memory_eval.py

memory-qa: ## long-form-QA RAG-vs-baseline measurement (ADVISORY, tools/memory_qa.py)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/memory_qa.py

rust-mem: ## build Rust astra-rt + astramem and run memory cross-check
	cargo build --release -p astra-rt --manifest-path service/rust/Cargo.toml
	$(PYTHON) tools/rust_mem_crosscheck.py

registry: ## show registered artifacts + verify checksums
	$(PYTHON) tools/registry.py list

serve: ## run the Python inference HTTP service on :8080
	$(PYTHON) service/inference.py --checkpoint checkpoints/name/resumed/final.npz \
	  --config configs/toy_name.json

rust-test: ## build + test the Rust runtime skeleton (astra-rt)
	cd service/rust && cargo test

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

test: ## run the full test suite
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) -m pytest tests/ -q

check: ## run tests (aliases: test)
	$(MAKE) test

tokenizer: ## train the Phase 0 tokenizer on toy data (EX-01)
	$(PYTHON) tokenizer/train_tokenizer.py --config configs/tokenizer.json

train: ## run toy pretraining (EX-03)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) training/train.py --config configs/toy_pretrain.json

eval: ## evaluation harness on the last checkpoint (EX-06)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) evaluation/evaluate.py \
	  --checkpoint checkpoints/phase0/final.npz --config configs/toy_pretrain.json

leak-check: ## cross-split n-gram contamination gate (EX-05)
	$(PYTHON) tools/leak_check.py --tokenizer tokenizer/artifacts/toy_bpe.json \
	  --train datasets/toy/train.txt --heldout datasets/toy/val.txt --n 13

param-count: ## parameter count for the Phase 0 model
	$(PYTHON) tools/param_count.py --config configs/toy_pretrain.json

generate: ## interactive text generation (prompt Astra in terminal)
	$(PYTHON) inference/generate.py \
	  --checkpoint checkpoints/name/resumed/final.npz \
	  --config configs/toy_name.json

birth-test: ## run the Astra 0.1 Birth Test (full pipeline verification)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/birth_test.py

experiments: ## query the recorded experiment store
	$(PYTHON) tools/experiments.py list

decontaminate: ## filter split docs sharing a 13-gram with train (hygiene step)
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/val.txt tokenizer/artifacts/toy_bpe.json
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/eval.txt tokenizer/artifacts/toy_bpe.json