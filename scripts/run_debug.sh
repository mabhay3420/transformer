./scripts/build_debug.sh
time ./build/debug/tformer
source .venv/bin/activate
if [ "$1" == "v" ];then
	python scripts/vis_loss.py
fi
