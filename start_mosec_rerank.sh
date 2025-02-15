mkdir -p rerank_logs

numactl -C 0-15 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8000 > rerank_logs/8000.logs 2>&1 &
numactl -C 16-31 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8001 > rerank_logs/8001.logs 2>&1 &
numactl -C 32-47 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8002 > rerank_logs/8002.logs 2>&1 &
numactl -C 48-63 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8003 > rerank_logs/8003.logs 2>&1 &

# numactl -C 64-79 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8004 > rerank_logs/8004.logs 2>&1 &
# numactl -C 79-94 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8005 > rerank_logs/8005.logs 2>&1 &
# numactl -C 95-111 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8006 > rerank_logs/8006.logs 2>&1 &
# numactl -C 112-127 -l python3 mosec_rerank_server.py --timeout=3000000 --port=8007 > rerank_logs/8007.logs 2>&1 &

