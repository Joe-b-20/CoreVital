#!/usr/bin/env bash
# GPU and CPU usage — uses only tools available on RunPod image:
#   nvidia-smi (CUDA), free, /proc/loadavg (standard Linux)

echo ""
echo "=============================================="
echo "  GPU"
echo "=============================================="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while read -r line; do
    echo "  $line"
  done
  nvidia-smi 2>/dev/null || true
else
  echo "  nvidia-smi not found"
fi

echo ""
echo "=============================================="
echo "  CPU & Memory"
echo "=============================================="
if [ -r /proc/loadavg ]; then
  read -r load1 load5 load15 _ _ < /proc/loadavg
  echo "  Load average: $load1 $load5 $load15 (1m 5m 15m)"
fi
echo ""
free -h 2>/dev/null || true
echo ""
