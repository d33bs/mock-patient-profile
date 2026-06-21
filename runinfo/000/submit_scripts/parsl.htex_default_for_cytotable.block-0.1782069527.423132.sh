
export JOBNAME=parsl.htex_default_for_cytotable.block-0.1782069527.423132
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py   -a 127.0.0.1,70.58.19.112,192.168.0.186 -p 0 -c 1.0 -m None --poll 10 --port=54378 --cert_dir /Users/buntend/Documents/work/mock-patient-profile/runinfo/000/htex_default_for_cytotable/certificates --logdir=/Users/buntend/Documents/work/mock-patient-profile/runinfo/000/htex_default_for_cytotable --logconf=None --block_id=0 --hb_period=30  --hb_threshold=120 --drain_period=None --cpu-affinity none  --mpi-launcher=mpiexec --available-accelerators
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
