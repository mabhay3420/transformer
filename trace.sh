./build_debug.sh
sudo rm -rf tformer_trace.trace 
codesign --entitlements entitlements.plist --sign - --force ./build/tformer
if [ "$1" = "t" ]; then
	xctrace record --template "Time Profiler" --output tformer_trace.trace --launch ./build/tformer
elif [ "$1" = "m" ]; then
	sudo xctrace record --template "Allocations" --output tformer_trace.trace --launch ./build/tformer
else 
	echo "Usage: ./build_debug.sh t (for time profiler) | m (for memory allocations)"
	exit 1
fi
sudo chmod -R a+rX tformer_trace.trace
sudo open tformer_trace.trace

