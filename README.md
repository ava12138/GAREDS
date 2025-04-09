# pro
manage my work in QG&amp;DG
# instruction
nohup python run-api.py -d scienceqa -m qwen7b -p rule 2>&1 | tee log/run-api.log &

nohup python run-api-vl.py -d scienceqa -m qwenvl -p rule 2>&1 | tee log/run-api-vl-fact.log &

nohup python run.py -d scienceqa -m qwenvl -p rule -i pt -g 7 > log/run-api-vl-local.log 2>&1 &
# evaluate
python eval-new.py -d scienceqa -m qwen7b -p rule
