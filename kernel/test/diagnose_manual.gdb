# CUDA-GDB Manual Diagnostic Script for b200_fp4 Attention Kernel
# After kernel hangs and you Ctrl+C, run: source diagnose_manual.gdb

set pagination off

# ==================================================
# CONFIGURE THIS: Which cluster to inspect (0, 1, 2, ...)
# ==================================================
set $cluster = 0

# Calculate which CTAs to check (cluster * 2 and cluster * 2 + 1)
set $cta0 = $cluster * 2
set $cta1 = $cluster * 2 + 1

printf "\n========================================\n"
printf "Inspecting Cluster %d (CTAs %d and %d)\n", $cluster, $cta0, $cta1
printf "========================================\n"

echo \n=== Deadlock Analysis ===\n\n

echo \n--- Active Kernels ---\n
info cuda kernels

cuda kernel 0

echo \n--- Active Blocks ---\n
info cuda blocks

echo \n--- Warp States ---\n
info cuda warps

echo \n=== Bitfield State Analysis ===\n

printf "\n--- CTA %d Bitfield Values ---\n", $cta0
eval "cuda block %d thread (0,0,0)", $cta0
frame function fwd_attend_ker
printf "CTA %d bitfield       = 0x%08X\n", $cta0, bitfield
printf "CTA %d scale_bitfield = 0x%08X\n", $cta0, scale_bitfield

printf "\n--- CTA %d Bitfield Interpretation (Lower 16 bits = arrived phases, Upper 16 bits = finished phases) ---\n", $cta0
printf "bitfield bits 0-15  (***_arrived): "
print /t bitfield & 0xFFFF
printf "bitfield bits 16-31 (***_finished): "
print /t (bitfield >> 16) & 0xFFFF

printf "\nscale_bitfield bits 0-15  (***_arrived): "
print /t scale_bitfield & 0xFFFF
printf "scale_bitfield bits 16-31 (***_finished): "
print /t (scale_bitfield >> 16) & 0xFFFF

printf "\n--- CTA %d Decoded Phase Bits ---\n", $cta0
printf "Ring buffers: [0] [1]\n"
printf "k_smem_arrived expected phases (bits 0-1):       [%d] [%d]\n", (bitfield & 1), ((bitfield >> 1) & 1)
printf "v_smem_arrived expected phases (bits 2-3):       [%d] [%d]\n", ((bitfield >> 2) & 1), ((bitfield >> 3) & 1)
printf "k_smem_finished expected phases (bits 16-17):    [%d] [%d]\n", ((bitfield >> 16) & 1), ((bitfield >> 17) & 1)
printf "v_smem_finished expected phases (bits 18-19):    [%d] [%d]\n", ((bitfield >> 18) & 1), ((bitfield >> 19) & 1)

printf "\nks_smem_arrived expected phases (scale bits 0-1):    [%d] [%d]\n", (scale_bitfield & 1), ((scale_bitfield >> 1) & 1)
printf "ks_smem_finished expected phases (scale bits 16-17): [%d] [%d]\n", ((scale_bitfield >> 16) & 1), ((scale_bitfield >> 17) & 1)

printf "\n--- CTA %d Bitfield Values ---\n", $cta1
eval "cuda block %d thread (0,0,0)", $cta1
frame function fwd_attend_ker
printf "CTA %d bitfield       = 0x%08X\n", $cta1, bitfield
printf "CTA %d scale_bitfield = 0x%08X\n", $cta1, scale_bitfield

printf "\n--- CTA %d Bitfield Interpretation (Lower 16 bits = arrived phases, Upper 16 bits = finished phases) ---\n", $cta1
printf "bitfield bits 0-15  (***_arrived): "
print /t bitfield & 0xFFFF
printf "bitfield bits 16-31 (***_finished): "
print /t (bitfield >> 16) & 0xFFFF

printf "\nscale_bitfield bits 0-15  (***_arrived): "
print /t scale_bitfield & 0xFFFF
printf "scale_bitfield bits 16-31 (***_finished): "
print /t (scale_bitfield >> 16) & 0xFFFF

printf "\n--- CTA %d Decoded Phase Bits ---\n", $cta1
printf "Ring buffers: [0] [1]\n"
printf "k_smem_arrived expected phases (bits 0-1):       [%d] [%d]\n", (bitfield & 1), ((bitfield >> 1) & 1)
printf "v_smem_arrived expected phases (bits 2-3):       [%d] [%d]\n", ((bitfield >> 2) & 1), ((bitfield >> 3) & 1)
printf "k_smem_finished expected phases (bits 16-17):    [%d] [%d]\n", ((bitfield >> 16) & 1), ((bitfield >> 17) & 1)
printf "v_smem_finished expected phases (bits 18-19):    [%d] [%d]\n", ((bitfield >> 18) & 1), ((bitfield >> 19) & 1)

printf "\nks_smem_arrived expected phases (scale bits 0-1):    [%d] [%d]\n", (scale_bitfield & 1), ((scale_bitfield >> 1) & 1)
printf "ks_smem_finished expected phases (scale bits 16-17): [%d] [%d]\n", ((scale_bitfield >> 16) & 1), ((scale_bitfield >> 17) & 1)

echo \n=== Semaphore State Analysis ===\n
echo Note: mbarrier semaphores are 64-bit opaque values in shared memory.\n
echo       Their internal state (phase, count) cannot be directly inspected.\n
echo       The phase bit we track is in the bitfield, not in the semaphore.\n
echo       Listed below are shared memory addresses for reference only.\n

printf "\n--- CTA %d Semaphore Shared Memory Addresses ---\n", $cta0
eval "cuda block %d thread (0,0,0)", $cta0
frame function fwd_attend_ker
printf "q_smem_arrived[0] @ 0x%x\n", &q_smem_arrived[0]
printf "q_smem_arrived[1] @ 0x%x\n", &q_smem_arrived[1]
printf "qs_smem_arrived @ 0x%x\n", &qs_smem_arrived
printf "qs_tmem_arrived @ 0x%x\n", &qs_tmem_arrived
printf "k_smem_arrived[0] @ 0x%x, k_smem_finished[0] @ 0x%x\n", &k_smem_arrived[0], &k_smem_finished[0]
printf "k_smem_arrived[1] @ 0x%x, k_smem_finished[1] @ 0x%x\n", &k_smem_arrived[1], &k_smem_finished[1]
printf "v_smem_arrived[0] @ 0x%x, v_smem_finished[0] @ 0x%x\n", &v_smem_arrived[0], &v_smem_finished[0]
printf "v_smem_arrived[1] @ 0x%x, v_smem_finished[1] @ 0x%x\n", &v_smem_arrived[1], &v_smem_finished[1]
printf "ks_smem_arrived[0] @ 0x%x, ks_smem_finished[0] @ 0x%x\n", &ks_smem_arrived[0], &ks_smem_finished[0]
printf "ks_smem_arrived[1] @ 0x%x, ks_smem_finished[1] @ 0x%x\n", &ks_smem_arrived[1], &ks_smem_finished[1]
printf "attn_unloaded[0] @ 0x%x, attn_mma_stored[0] @ 0x%x\n", &attn_unloaded[0], &attn_mma_stored[0]
printf "attn_unloaded[1] @ 0x%x, attn_mma_stored[1] @ 0x%x\n", &attn_unloaded[1], &attn_mma_stored[1]
printf "qk_matmul_done[0] @ 0x%x, av_matmul_done[0] @ 0x%x\n", &qk_matmul_done[0], &av_matmul_done[0]
printf "qk_matmul_done[1] @ 0x%x, av_matmul_done[1] @ 0x%x\n", &qk_matmul_done[1], &av_matmul_done[1]

printf "\n--- CTA %d Semaphore Shared Memory Addresses ---\n", $cta1
eval "cuda block %d thread (0,0,0)", $cta1
frame function fwd_attend_ker
printf "q_smem_arrived[0] @ 0x%x\n", &q_smem_arrived[0]
printf "q_smem_arrived[1] @ 0x%x\n", &q_smem_arrived[1]
printf "qs_smem_arrived @ 0x%x\n", &qs_smem_arrived
printf "qs_tmem_arrived @ 0x%x\n", &qs_tmem_arrived
printf "k_smem_arrived[0] @ 0x%x, k_smem_finished[0] @ 0x%x\n", &k_smem_arrived[0], &k_smem_finished[0]
printf "k_smem_arrived[1] @ 0x%x, k_smem_finished[1] @ 0x%x\n", &k_smem_arrived[1], &k_smem_finished[1]
printf "v_smem_arrived[0] @ 0x%x, v_smem_finished[0] @ 0x%x\n", &v_smem_arrived[0], &v_smem_finished[0]
printf "v_smem_arrived[1] @ 0x%x, v_smem_finished[1] @ 0x%x\n", &v_smem_arrived[1], &v_smem_finished[1]
printf "ks_smem_arrived[0] @ 0x%x, ks_smem_finished[0] @ 0x%x\n", &ks_smem_arrived[0], &ks_smem_finished[0]
printf "ks_smem_arrived[1] @ 0x%x, ks_smem_finished[1] @ 0x%x\n", &ks_smem_arrived[1], &ks_smem_finished[1]
printf "attn_unloaded[0] @ 0x%x, attn_mma_stored[0] @ 0x%x\n", &attn_unloaded[0], &attn_mma_stored[0]
printf "attn_unloaded[1] @ 0x%x, attn_mma_stored[1] @ 0x%x\n", &attn_unloaded[1], &attn_mma_stored[1]
printf "qk_matmul_done[0] @ 0x%x, av_matmul_done[0] @ 0x%x\n", &qk_matmul_done[0], &av_matmul_done[0]
printf "qk_matmul_done[1] @ 0x%x, av_matmul_done[1] @ 0x%x\n", &qk_matmul_done[1], &av_matmul_done[1]

echo \n=== Loop Iteration Variables ===\n

printf "\n--- CTA %d ---\n", $cta0

printf "\n--- CTA %d Consumer 0 (Block %d, Thread 0) ---\n", $cta0, $cta0
eval "cuda block %d thread (0,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"k_input_ring = %d\\n\", k_input_ring")
except:
    print("k_input_ring = <not in scope>")
try:
    gdb.execute("printf \"v_input_ring = %d\\n\", v_input_ring")
except:
    print("v_input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d Consumer 1 (Block %d, Thread 256) ---\n", $cta0, $cta0
eval "cuda block %d thread (256,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"k_input_ring = %d\\n\", k_input_ring")
except:
    print("k_input_ring = <not in scope>")
try:
    gdb.execute("printf \"v_input_ring = %d\\n\", v_input_ring")
except:
    print("v_input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d QK MMA Launcher (Block %d, Thread 512) ---\n", $cta0, $cta0
eval "cuda block %d thread (512,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"scale_ring = %d\\n\", scale_ring")
except:
    print("scale_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d AV MMA Launcher (Block %d, Thread 544) ---\n", $cta0, $cta0
eval "cuda block %d thread (544,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d K Loader (Block %d, Thread 576) ---\n", $cta0, $cta0
eval "cuda block %d thread (576,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"scale_ring = %d\\n\", scale_ring")
except:
    print("scale_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d V Loader (Block %d, Thread 608) ---\n", $cta0, $cta0
eval "cuda block %d thread (608,0,0)", $cta0
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d ---\n", $cta1

printf "\n--- CTA %d Consumer 0 (Block %d, Thread 0) ---\n", $cta1, $cta1
eval "cuda block %d thread (0,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"k_input_ring = %d\\n\", k_input_ring")
except:
    print("k_input_ring = <not in scope>")
try:
    gdb.execute("printf \"v_input_ring = %d\\n\", v_input_ring")
except:
    print("v_input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d Consumer 1 (Block %d, Thread 256) ---\n", $cta1, $cta1
eval "cuda block %d thread (256,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"k_input_ring = %d\\n\", k_input_ring")
except:
    print("k_input_ring = <not in scope>")
try:
    gdb.execute("printf \"v_input_ring = %d\\n\", v_input_ring")
except:
    print("v_input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d QK MMA Launcher (Block %d, Thread 512) ---\n", $cta1, $cta1
eval "cuda block %d thread (512,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"scale_ring = %d\\n\", scale_ring")
except:
    print("scale_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d AV MMA Launcher (Block %d, Thread 544) ---\n", $cta1, $cta1
eval "cuda block %d thread (544,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d K Loader (Block %d, Thread 576) ---\n", $cta1, $cta1
eval "cuda block %d thread (576,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"scale_ring = %d\\n\", scale_ring")
except:
    print("scale_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

printf "\n--- CTA %d V Loader (Block %d, Thread 608) ---\n", $cta1, $cta1
eval "cuda block %d thread (608,0,0)", $cta1
frame function fwd_attend_ker
python
try:
    gdb.execute("printf \"input_ring = %d\\n\", input_ring")
except:
    print("input_ring = <not in scope>")
try:
    gdb.execute("printf \"task_iter = %d\\n\", task_iter")
except:
    print("task_iter = <not in scope>")
try:
    gdb.execute("printf \"idx = %d\\n\", idx")
except:
    print("idx = <not in scope>")
end

echo \n=== Thread Backtraces ===\n

printf "\n--- CTA %d Threads ---\n", $cta0

printf "\n--- CTA %d Consumer 0 (Block %d, Thread 0) ---\n", $cta0, $cta0
eval "cuda block %d thread (0,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d Consumer 1 (Block %d, Thread 256) ---\n", $cta0, $cta0
eval "cuda block %d thread (256,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d QK MMA Launcher (Block %d, Thread 512) ---\n", $cta0, $cta0
eval "cuda block %d thread (512,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d AV MMA Launcher (Block %d, Thread 544) ---\n", $cta0, $cta0
eval "cuda block %d thread (544,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d K Loader (Block %d, Thread 576) ---\n", $cta0, $cta0
eval "cuda block %d thread (576,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d V Loader (Block %d, Thread 608) ---\n", $cta0, $cta0
eval "cuda block %d thread (608,0,0)", $cta0
bt 5
where
frame 0
list

printf "\n--- CTA %d Threads ---\n", $cta1

printf "\n--- CTA %d Consumer 0 (Block %d, Thread 0) ---\n", $cta1, $cta1
eval "cuda block %d thread (0,0,0)", $cta1
bt 5
where
frame 0
list

printf "\n--- CTA %d Consumer 1 (Block %d, Thread 256) ---\n", $cta1, $cta1
eval "cuda block %d thread (256,0,0)", $cta1
bt 5
where
frame 0
list

printf "\n--- CTA %d QK MMA Launcher (Block %d, Thread 512) ---\n", $cta1, $cta1
eval "cuda block %d thread (512,0,0)", $cta1
bt 5
where
frame 0
list

printf "\n--- CTA %d AV MMA Launcher (Block %d, Thread 544) ---\n", $cta1, $cta1
eval "cuda block %d thread (544,0,0)", $cta1
bt 5
where
frame 0
list

printf "\n--- CTA %d K Loader (Block %d, Thread 576) ---\n", $cta1, $cta1
eval "cuda block %d thread (576,0,0)", $cta1
bt 5
where
frame 0
list

printf "\n--- CTA %d V Loader (Block %d, Thread 608) ---\n", $cta1, $cta1
eval "cuda block %d thread (608,0,0)", $cta1
bt 5
where
frame 0
list

echo \n=== Analysis Complete ===\n
echo \n
echo ========================================\n
echo INTERPRETATION GUIDE FOR b200_fp4.cu\n
echo ========================================\n
echo \n
echo Bitfield Layout (uint32_t bitfield = 0xFFFF0000):\n
echo   - Bits 0-15:  Expected phases for ***_arrived barriers\n
echo   - Bits 16-31: Expected phases for ***_finished barriers\n
echo \n
echo Phase Bit Mapping for bitfield:\n
echo   - Bits 0-1:   k_smem_arrived[0-1]\n
echo   - Bits 2-3:   v_smem_arrived[0-1]\n
echo   - Bits 4-15:  attn_unloaded, attn_mma_stored, qk/av_matmul_done\n
echo   - Bits 16-17: k_smem_finished[0-1]\n
echo   - Bits 18-19: v_smem_finished[0-1]\n
echo   - Bits 20-31: attn completion barriers\n
echo \n
echo Phase Bit Mapping for scale_bitfield:\n
echo   - Bits 0-1:   ks_smem_arrived[0-1]\n
echo   - Bits 16-17: ks_smem_finished[0-1]\n
echo   Note: K scales are loaded to SMEM, then copied to TMEM inline\n
echo         No separate ks_tmem semaphores exist\n
echo \n
echo Common Deadlock Patterns:\n
echo   1. Producer waiting on ***_finished, consumer waiting on ***_arrived\n
echo      -> Check if phase bits match what barrier is actually at\n
echo \n
echo   2. Mismatched iteration counters (input_ring, scale_ring)\n
echo      -> Producer and consumer ring positions should differ by ~1 stage\n
echo \n
echo   3. One CTA progressed, other CTA stuck\n
echo      -> Check cluster synchronization (tma::cluster::wait calls)\n
echo      -> Check multicast arrive counts\n
echo \n
echo   4. All threads stuck at tma::cluster::wait\n
echo      -> Check which barrier and phase they're waiting for\n
echo      -> Cross-reference with producer thread state\n
echo \n
echo Look for threads stuck at tma::cluster::wait() or kittens::wait()\n
echo Compare their expected phase (from bitfield) with producer progress\n
echo \n