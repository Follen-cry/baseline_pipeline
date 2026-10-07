#!/bin/bash
# Orchestrates the 4 independent S0-S3 gen-SFT trainings across torrnode8's
# GPUs, per the 2026-08-31 resource-allocation plan (torrnode8 had 7/8 A40s
# free at the time -- idx1 was the one busy GPU, see data/specs/S0_S3_DATASET_SPEC.md
# era planning conversation):
#
#   - Block A (GPUs 0,2,3,4) starts training immediately with the first
#     queued setting.
#   - Block B (GPUs 1,5,6,7) is polled every POLL_INTERVAL seconds; the next
#     queued setting is launched there ONLY once all 4 of its GPUs are
#     simultaneously idle (<500MiB used) -- i.e. once whatever's holding idx1
#     finishes.
#   - Whenever a block's current job finishes (its run_s{N}_gen_sft.sh
#     process -- train + processor-fix + LoRA-merge + verify -- exits), the
#     next not-yet-launched setting in the queue is started on that block.
#   - Total torrnode8 usage is capped at these 8 GPUs (0-7) -- never more.
#   - If Block B never frees up, all 4 settings simply run sequentially on
#     Block A alone (this is the intended fallback, not an error case).
#
# This script is meant to run fully detached on torrnode8 itself (launched
# via `nohup ... & disown` over ssh), so it survives the launching session
# ending -- it does not depend on any outside process staying alive.
#
# Usage:
#   ssh torrnode8
#   cd /scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat
#   nohup bash shell/internvlu/orchestration/v1/orchestrate_s0s3_torrnode8.sh \
#     > logs/s0s3_torrnode8/orchestrator.log 2>&1 < /dev/null &
#   disown
# Or, one-shot from any node:
#   ssh torrnode8 'cd /scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat && mkdir -p logs/s0s3_torrnode8 && nohup bash shell/internvlu/orchestration/v1/orchestrate_s0s3_torrnode8.sh > logs/s0s3_torrnode8/orchestrator.log 2>&1 < /dev/null & disown'
set -uo pipefail

REPO="/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat"
LOGDIR="$REPO/logs/s0s3_torrnode8"
mkdir -p "$LOGDIR"
cd "$REPO"

CONDA_SH="/homes/55/junlin/miniconda3/etc/profile.d/conda.sh"

QUEUE=(s1 s2 s3)
NEXT=0

BLOCK_A_GPUS="0,2,3,4"
BLOCK_B_GPUS="1,5,6,7"
BLOCK_B_IDX=(1 5 6 7)
FREE_THRESHOLD_MIB=500   # a GPU with less than this used is "idle"

A_PIDFILE=""
B_PIDFILE=""
A_SETTING=""
B_SETTING=""

POLL_INTERVAL=180   # seconds between Block-B-availability / job-liveness checks

log() { echo "$(date '+%F %T') [orchestrator] $*" >&2; }
# ^ NOTE: must go to stderr, not stdout -- launch() below is called as
# A_PIDFILE=$(launch ...), and command substitution captures ALL of a
# function's stdout. If log() wrote to stdout, its "launching ..." line
# would get glued onto the returned pidfile path, corrupting A_PIDFILE/
# B_PIDFILE into a garbled multi-line string that job_alive()'s `[ -f "$1" ]`
# check always fails against -- making every job look "finished" one poll
# cycle after it starts, regardless of whether it's actually still running.
# (Bug found and fixed 2026-08-31 after it caused exactly this: the real S0
# job kept training fine, but s1/s2/s3 got spuriously launched on top of the
# still-running Block A/B GPUs and crashed within seconds, and the queue got
# marked fully "dispatched" without any of them actually completing.)

job_alive() {  # $1 = pidfile
  [ -f "$1" ] || return 1
  local pid; pid=$(cat "$1" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

launch() {
  local setting=$1 gpus=$2 block=$3
  local ngpu; ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
  local logf="$LOGDIR/${setting}_block${block}_$(date +%Y%m%d_%H%M%S).log"
  local pidfile="$LOGDIR/${setting}_block${block}.pid"
  local exitfile="$LOGDIR/${setting}_block${block}.exit"
  rm -f "$exitfile"
  log "launching $setting on block $block (GPUs $gpus, ${ngpu}-way DDP) -> $logf"
  (
    source "$CONDA_SH"
    conda activate internvlu
    cd "$REPO"
    # Inner subshell wraps the actual run so its real exit code (train +
    # processor-fix + LoRA-merge + verify all succeeded, vs crashed partway)
    # gets recorded -- job_alive() alone only tells us the process is gone,
    # not whether it finished cleanly, which matters for an unattended
    # multi-day run through all 4 settings.
    (
      CUDA_VISIBLE_DEVICES="$gpus" GPUS="$ngpu" \
        nohup bash "shell/internvlu/sft/run_${setting}_gen_sft.sh" > "$logf" 2>&1 < /dev/null
      echo $? > "$exitfile"
    ) &
    echo $! > "$pidfile"
  )
  echo "$pidfile"
}

block_b_free() {
  for i in "${BLOCK_B_IDX[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null)
    [ -z "$used" ] && return 1
    [ "$used" -ge "$FREE_THRESHOLD_MIB" ] && return 1
  done
  return 0
}

log "starting. queue=${QUEUE[*]}  block_a_gpus=$BLOCK_A_GPUS  block_b_gpus=$BLOCK_B_GPUS  poll=${POLL_INTERVAL}s"

# ---- kick off Block A with the first queued setting, UNLESS it's already
# running (e.g. this orchestrator is being restarted while a prior instance's
# launch of it is still alive) -- adopt it instead of double-launching a
# conflicting duplicate on the same GPUs ----
FIRST_SETTING="${QUEUE[$NEXT]}"
FIRST_PIDFILE="$LOGDIR/${FIRST_SETTING}_blockA.pid"
if job_alive "$FIRST_PIDFILE"; then
  log "resuming: $FIRST_SETTING already running on block A (pid $(cat "$FIRST_PIDFILE")) -- adopting, not relaunching"
  A_SETTING="$FIRST_SETTING"
  A_PIDFILE="$FIRST_PIDFILE"
  NEXT=$((NEXT + 1))
else
  A_SETTING="$FIRST_SETTING"
  A_PIDFILE=$(launch "$A_SETTING" "$BLOCK_A_GPUS" "A")
  NEXT=$((NEXT + 1))
fi

while :; do
  sleep "$POLL_INTERVAL"

  # queue exhausted and both blocks idle -> we're done
  if [ "$NEXT" -ge "${#QUEUE[@]}" ] && [ -z "$A_SETTING" ] && [ -z "$B_SETTING" ]; then
    log "all ${#QUEUE[@]} settings launched and finished:"
    for s in "${QUEUE[@]}"; do
      for b in A B; do
        f="$LOGDIR/${s}_block${b}.exit"
        [ -f "$f" ] && log "  $s (block $b): exit=$(cat "$f")"
      done
    done
    break
  fi

  # ---- Block A: check completion, refill from queue ----
  if [ -n "$A_SETTING" ] && ! job_alive "$A_PIDFILE"; then
    rc=$(cat "$LOGDIR/${A_SETTING}_blockA.exit" 2>/dev/null || echo "?")
    log "block A finished: $A_SETTING (exit=$rc)$([ "$rc" != 0 ] && echo ' *** NON-ZERO EXIT -- CHECK LOG ***')"
    A_SETTING=""
  fi
  if [ -z "$A_SETTING" ] && [ "$NEXT" -lt "${#QUEUE[@]}" ]; then
    A_SETTING="${QUEUE[$NEXT]}"
    A_PIDFILE=$(launch "$A_SETTING" "$BLOCK_A_GPUS" "A")
    NEXT=$((NEXT + 1))
  fi

  # ---- Block B: check completion ----
  if [ -n "$B_SETTING" ] && ! job_alive "$B_PIDFILE"; then
    rc=$(cat "$LOGDIR/${B_SETTING}_blockB.exit" 2>/dev/null || echo "?")
    log "block B finished: $B_SETTING (exit=$rc)$([ "$rc" != 0 ] && echo ' *** NON-ZERO EXIT -- CHECK LOG ***')"
    B_SETTING=""
  fi

  # ---- Block B: opportunistically start next queued setting if free ----
  if [ -z "$B_SETTING" ] && [ "$NEXT" -lt "${#QUEUE[@]}" ] && block_b_free; then
    B_SETTING="${QUEUE[$NEXT]}"
    B_PIDFILE=$(launch "$B_SETTING" "$BLOCK_B_GPUS" "B")
    NEXT=$((NEXT + 1))
  fi
done

log "orchestrator done."
