.PHONY: check check-strict check-fast

check:
	python scripts/local_check.py

check-strict:
	python scripts/local_check.py --strict-physics --strict-occlusion

check-fast:
	python scripts/local_check.py --skip-tests --keep-output
