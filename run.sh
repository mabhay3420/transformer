
./build.sh
time ./build/release/tformer
source .venv/bin/activate
if [ "$1" == "v" ]; then
    python vis_loss.py
elif [ "$1" == "v2" ]; then
    python vis_loss_v2.py
elif [ "$1" == "v3" ]; then
    python vis_loss_v3.py
fi
