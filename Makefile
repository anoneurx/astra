PYTHON ?= python3
THREADS ?= 8

.PHONY: test check tokenizer train eval leak param-count help

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

decontaminate: ## filter split docs sharing a 13-gram with train (hygiene step)
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/val.txt tokenizer/artifacts/toy_bpe.json
	$(PYTHON) datasets/toy/corpus.py --decontaminate \
	  datasets/toy/train.txt datasets/toy/eval.txt tokenizer/artifacts/toy_bpe.json