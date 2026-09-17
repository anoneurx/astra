PYTHON ?= python3
THREADS ?= 8
VERSION ?= 1.0.0

.PHONY: test check tokenizer train eval leak param-count generate chat birth-test experiments decontaminate registry serve rust-test benchmark memory-eval memory-qa rust-mem learning-loop self-improve rollback-drill e2e release lint typecheck word-tokenizer word-prose word-chat word-train

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

learning-loop: ## run the Phase-5 candidate learning loop (validated feedback -> candidate)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/learning_loop.py

self-improve: ## run the Phase-6 supervised self-improvement loop (gate -> promote/reject -> audit)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/self_improve.py

self-learn: ## run ONE self-learning cycle off the drive daemon (feedback -> candidate -> gate -> promote)
	OPENBLAS_NUM_THREADS=$(THREADS) REPO=$(PWD) $(PYTHON) \
	  /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/selflearn/selflearn_runner.py --once

self-learn-daemon: ## start the continuous self-learning daemon (watch feedback forever)
	OPENBLAS_NUM_THREADS=$(THREADS) REPO=$(PWD) $(PYTHON) \
	  /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/selflearn/selflearn_runner.py

rollback-drill: ## offline auto-rollback drill (promote regressing artifact, restore prior sha)
	$(PYTHON) tools/self_improve.py --rollback-drill

e2e: ## Phase-7 full-stack walk: tokenizer -> registry -> model -> inference -> memory -> learning -> gates -> promote -> audit
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/e2e.py

release: ## Phase-7 release-checklist automation: full gates before tagging (docs/RELEASES.md § 2)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/release.py check --version $(VERSION)

registry: ## show registered artifacts + verify checksums
	$(PYTHON) tools/registry.py list

serve: ## run the Python inference HTTP service on :8080
	$(PYTHON) service/inference.py --checkpoint checkpoints/name/resumed/final.npz \
	  --config configs/toy_name.json

rust-test: ## build + test the Rust runtime skeleton (astra-rt)
	cd service/rust && cargo test

lint: ## ruff check (fast static lint); prefers the repo .venv ruff
	$(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff) check python/ tools/ tests/

typecheck: ## mypy on astra (type-check); uses .venv mypy if present
	$(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,mypy) python/astra --ignore-missing-imports

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

generate: ## interactive text generation (prompt Astra in terminal; auto-uses best language model)
	$(PYTHON) inference/generate.py

chat: ## interactive chat using the prose-chat fine-tune (Astra speaks native English in dialogue form)
	$(PYTHON) inference/generate.py --chat

word-tokenizer: ## build the 16k word-level tokenizer over prose + chat corpora (native-English fix)
	$(PYTHON) tools/build_word_tokenizer.py

word-prose: ## train the word-level English base (8.1M, datasets/prose) via the drive chunk runner
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) \
	  /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/chunk_runner_word_prose.py

word-chat: ## fine-tune the word-level model on the distilled chat corpus (drive chunk runner)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) \
	  /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/chunk_runner_word_chat.py

word-train: ## train BOTH word-level stages end-to-end: prose base then chat fine-tune
	$(MAKE) word-prose
	$(MAKE) word-chat

birth-test: ## run the Astra 0.1 Birth Test (full pipeline verification)
	OPENBLAS_NUM_THREADS=$(THREADS) $(PYTHON) tools/birth_test.py

experiments: ## query the recorded experiment store
	$(PYTHON) tools/experiments.py list

decontaminate: ## filter split docs sharing a 13-gram with train (hygiene step)
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/val.txt tokenizer/artifacts/toy_bpe.json
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/eval.txt tokenizer/artifacts/toy_bpe.json