./build_release.sh
time ./build/release/tformer
source .venv/bin/activate
if [ "$1" == "v" ];then
	python vis_loss.py
fi
