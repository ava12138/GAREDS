# pro
manage my work in QG&amp;DG
# instruction
PYTHONIOENCODING=utf8 && nohup python run-api.py -d scienceqa -m qwen7b -p rule 2>&1 | tee log/run-api.log &

PYTHONIOENCODING=utf8 && nohup python run.py -d scienceqa -m qwenvl -p rule -i pt -g 7 > log/run-api-vl-local.log 2>&1 &

python run.py -d scienceqa -m qwen7b -i vllm -p rule -g 7 --split validation
# evaluate
python eval.py -d scienceqa -m qwen7b -p rule --split test -w api
python eval-new.py -d scienceqa -i ./output/output_dg-scienceqa-coe-img-cot-test.json -o ./evaluation/scienceqa-coevl-cot-test.json --split test
transformers                      4.49.0
ms-swift  3.1.0