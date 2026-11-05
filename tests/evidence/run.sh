#!/bin/bash
# G8 final: port scores at 1.8M on GPU, then the original evaluators on CPU.
#
# Claim checked: policies the port trained (G8a seeds 1 to 3 at 200k, 300k and 1.8M grad steps)
# score the same in the port's evaluator and in the unmodified original's, and the reference
# checkpoint scores the same in both. Port side: roboballet_mjlab.scripts.eval on the training
# distribution and the paper tasksets. Original side: pt_to_flat.py, then orig_train.py and
# orig_static.py in the old venv, 24 or 48 CPU processes per checkpoint, rows pooled afterwards.
#
# Set before running:
#   ROBOBALLET_PROJECT  holds the original's source tree as roboballet/ and its venv as venv/
#   RUNS                holds the port's run directories g8a_s1 g8a_s2 g8a_s3 g8a_ref0 g8b_s1
#   OUT                 where the JSON and logs go
# Optional: REFERENCE_CKPT (default $ROBOBALLET_PROJECT/runs/mjlab_port_analysis/ckpt/paper_4r10t_s1_1870000.pt,
# as tests/golden/agent_load.py), ROBOBALLET_DATA (default data/ in this clone, as roboballet_mjlab.paths:
# the original's start states and the large pools), PYTHON for the release environment (default python),
# GPU (default 0), PIN_PORT and PIN_ORIGINAL for the CPU pinning the recorded run used.
: "${ROBOBALLET_PROJECT:?}" "${RUNS:?}" "${OUT:?}"
EVIDENCE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$EVIDENCE/../.." && pwd)
PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
PIN_PORT=${PIN_PORT:-taskset -c 64-71 nice -n 5}
PIN_ORIGINAL=${PIN_ORIGINAL:-taskset -c 0-47 nice -n 15}
REFERENCE_CKPT=${REFERENCE_CKPT:-$ROBOBALLET_PROJECT/runs/mjlab_port_analysis/ckpt/paper_4r10t_s1_1870000.pt}
DATA=${ROBOBALLET_DATA:-$REPO/data}
PB=$DATA/start_states/4r_10t_30o.pb
SCENES=$DATA/scenes  # The large pools. The small static ones are in the package:
PACKAGED=$REPO/src/roboballet_mjlab/data/scenes
STATIC=$PACKAGED/4r_10t_30o_static_paper.pt
mkdir -p $OUT/port $OUT/original
cd $REPO
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
EV="$PIN_PORT $PYTHON -m roboballet_mjlab.scripts.eval --worlds 4096 --static-worlds 1280 --seed 11"
for r in g8a_s1 g8a_s2 g8a_s3 g8a_ref0; do
  $EV --checkpoint $RUNS/$r/policies/policy_01800000.pt --arm-model mesh_tool --training-dist $PB --static-paper $STATIC --json $OUT/port/$r.json > $OUT/port/$r.log 2>&1
  echo "port $r exit $?"
done
$EV --checkpoint $RUNS/g8b_s1/policies/policy_01800000.pt --arm-model spheres_stick --training-dist $SCENES/4r_10t_30o_train.pt --heldout $SCENES/4r_10t_30o_heldout.pt --static-dist $PACKAGED/4r_10t_30o_static.pt --static-paper $STATIC --json $OUT/port/g8b_s1.json > $OUT/port/g8b_s1.log 2>&1
echo "port g8b_s1 exit $?"
$EV --checkpoint $REFERENCE_CKPT --arm-model mesh_tool --training-dist $PB --static-paper $STATIC --json $OUT/port/g4_reference.json > $OUT/port/g4_reference.log 2>&1
echo "port g4 exit $?"

for r in g8a_s1 g8a_s2 g8a_s3; do
  for s in 00200000 00300000 01800000; do
    $PYTHON $EVIDENCE/pt_to_flat.py $RUNS/$r/policies/policy_$s.pt $OUT/original/${r}_$s.npz
  done
done

cd $ROBOBALLET_PROJECT
unset OMP_NUM_THREADS
export PYTHONPATH=$ROBOBALLET_PROJECT PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export LD_LIBRARY_PATH="$(venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib"
PIN=$PIN_ORIGINAL
for r in g8a_s1 g8a_s2 g8a_s3; do
  W=$OUT/original/${r}_01800000.npz
  mkdir -p $OUT/original/${r}_final
  for i in $(seq 1 24); do
    $PIN venv/bin/python $EVIDENCE/orig_train.py $W 40 $((i + 300)) $OUT/original/${r}_final/train_$i.json > $OUT/original/${r}_final/train_$i.log 2>&1 &
    $PIN venv/bin/python $EVIDENCE/orig_static.py $W 5 $OUT/original/${r}_final/static_$i.json > $OUT/original/${r}_final/static_$i.log 2>&1 &
  done
  wait
  echo "original final $r done"
done
for r in g8a_s1 g8a_s2 g8a_s3; do
  for s in 00200000 00300000; do
    W=$OUT/original/${r}_$s.npz
    mkdir -p $OUT/original/${r}_$s
    for i in $(seq 1 48); do
      $PIN venv/bin/python $EVIDENCE/orig_train.py $W 20 $((i + 400)) $OUT/original/${r}_$s/train_$i.json > $OUT/original/${r}_$s/train_$i.log 2>&1 &
    done
    wait
    echo "original $r $s done"
  done
done

cd $REPO
unset JAX_PLATFORMS XLA_FLAGS LD_LIBRARY_PATH OPENBLAS_NUM_THREADS MKL_NUM_THREADS PYTHONDONTWRITEBYTECODE
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
for r in g8a_s1 g8a_s2 g8a_s3; do
  for s in 00200000 00300000; do
    $EV --checkpoint $RUNS/$r/policies/policy_$s.pt --arm-model mesh_tool --training-dist $PB --static-paper $STATIC --json $OUT/port/${r}_$s.json > $OUT/port/${r}_$s.log 2>&1
    echo "port $r $s exit $?"
  done
done

echo g8-final-evals-done
