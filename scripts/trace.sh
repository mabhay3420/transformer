TARGET=${2:-"xor-pt"}
./scripts/build_debug.sh
sudo rm -rf tformer_trace.trace ./build/debug/tformer.dSYM
xcrun dsymutil ./build/debug/tformer -o ./build/debug/tformer.dSYM
codesign --entitlements entitlements.plist --sign - --force ./build/debug/tformer
if [ "$1" = "t" ]; then
	xctrace record --template "Time Profiler" --output tformer_trace.trace --launch ./build/debug/tformer $TARGET
elif [ "$1" = "m" ]; then
	sudo xctrace record --template "Allocations" --output tformer_trace.trace --launch ./build/debug/tformer $TARGET
else
	echo "Usage: ./scripts/build_debug.sh t (for time profiler) | m (for memory allocations)"
	exit 1
fi
sudo chmod -R a+rX tformer_trace.trace
sudo open tformer_trace.trace
