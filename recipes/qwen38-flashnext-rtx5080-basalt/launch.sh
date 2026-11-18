#!/bin/bash
# Basalt Qwen3.8 FN on :8080. Usage: launch.sh [fast|exact]  (default fast)
MODE=${1:-fast}
B=${BASALT_DIR:-$HOME/basalt-0.4.1}; REL=$B/basalt-v0.4.1-linux-x64
P=$(ss -ltnp | grep ":8080 " | grep -o "pid=[0-9]*" | cut -d= -f2); [ -n "$P" ] && kill $P && sleep 10
pkill -f "bin/basalt --serve"; sleep 2
ARGS="--max-context 131072 --kv int8 --kv-resident 0 --prefill-cpu-share 0.4 --vision dynamic"
[ "$MODE" = fast ] && ARGS="$ARGS --relaxed-drafting 0.3 --route-prior 1.5 --route-tail-skip 7"
cd $REL && setsid nohup bin/basalt serve $B/Qwen3.8-FN-UD-IQ4_XS.basalt --python $B/.venv/bin/python \
  --host 0.0.0.0 --port 8080 --api-key "${BASALT_API_KEY:?set BASALT_API_KEY}" --alias qwen3.8-flash-next-v31-basalt \
  $ARGS > "${BASALT_LOG:-basalt-qwen.log}" 2>&1 < /dev/null &
echo "basalt $MODE starting"
