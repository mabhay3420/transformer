./build.sh
time ./build/tformer
source .venv/bin/activate
if [ "$1" == "v" ];then
	python vis_loss.py
fi
