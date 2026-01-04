#!/usr/bin/env bash

SCRIPTS_DIR=$(dirname "$0")

# Note to use Advanced Templates with different config options
# one can go to Instruments -> Choose a template -> Set configs
# File -> Save as Template
# And then specify the path to the template file
# e.g. TEMPLATE="${SCRIPTS_DIR}/cpu_counters_guided_template.tracetemplate"
# Alternatively, one can start a new run with different template
# inside an already opened trace session
if [ "$1" = "t" ]; then
	TEMPLATE="Time Profiler"
elif [ "$1" = "m" ]; then
	TEMPLATE="Allocations"
elif [ "$1" = "c" ]; then
	TEMPLATE="CPU Profiler"
elif [ "$1" = "cu" ]; then
	TEMPLATE="CPU Counters"
else

	echo "Usage: ./scripts/trace.sh {t|m|c|cu} [debug_binary_relpath] [args...]"
	echo "Note: debug_binary_relpath is relative to ./build/debug"
	echo "Example: ./scripts/trace.sh t microbenchmarks/adamw/adamw_bench_main"
	exit 1
fi

shift || true
BIN_REL=${1:-"tformer"}
if [ -n "${1:-}" ]; then
	shift || true
fi
EXTRA_ARGS=("$@")

BIN_PATH=$( realpath "./build/debug/$BIN_REL" )
if [ "$BIN_REL" = "tformer" ] && [ "${#EXTRA_ARGS[@]}" -eq 0 ]; then
	EXTRA_ARGS=("xor")
fi

BIN_NAME=$(basename "$BIN_PATH")
TRACE_OUTPUT="${BIN_NAME}_trace.trace"
DSYM_PATH="${BIN_PATH}.dSYM"

sudo rm -rf "$TRACE_OUTPUT" "$DSYM_PATH"
./scripts/build_debug.sh
xcrun dsymutil "$BIN_PATH" -o "$DSYM_PATH"
sudo codesign --entitlements entitlements.plist --sign - --force "$BIN_PATH"
xctrace record --template "$TEMPLATE" --output "$TRACE_OUTPUT" --launch "$BIN_PATH" "${EXTRA_ARGS[@]}"
# xctrace symbolicate --input "$TRACE_OUTPUT" --output "$TRACE_OUTPUT" --dsym "$DSYM_PATH"
sudo chmod -R a+rX "$TRACE_OUTPUT"
sudo open "$TRACE_OUTPUT"
