// A sampling profiler for one thread, for machines where perf is locked
// (perf_event_paranoid 4) and nsys CPU sampling is unavailable. Load it with LD_PRELOAD:
//
//   gcc -O2 -fPIC -shared tools/pmp/pmp.c -o /tmp/libpmp.so -lrt
//   PMP_OUT=/tmp/pmp.out PMP_US=250 LD_PRELOAD=/tmp/libpmp.so <command>
//   tools/pmp/pmp-report.py /tmp/pmp.out.<pid> 40 [t0:t1,...]
//
// It samples the CPU time of the thread that loads it (the main thread; for llama-server that
// is the thread running the decode loop) every PMP_US microseconds, and records the PC, up
// to DEPTH return addresses and a wall-clock stamp per sample. At exit it writes the samples and
// /proc/self/maps to $PMP_OUT.<pid>. Processes with fewer than 100 samples write nothing, so
// wrapping a script that spawns helpers is fine. Spin-waits in cudaStreamSynchronize show up
// as CPU time inside libcuda.
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define MAXS 400000
#define DEPTH 12

static void * samples[MAXS][DEPTH];
static volatile int nsamp = 0;
static double stamps[MAXS];
static timer_t tm;
static int active = 0;

static void handler(int sig, siginfo_t * si, void * uc_) {
    (void) sig; (void) si;
    int i = nsamp;
    if (i >= MAXS) return;
    ucontext_t * uc = (ucontext_t *) uc_;
    void * buf[DEPTH + 3];
    int n = backtrace(buf, DEPTH + 3);
    memset(samples[i], 0, sizeof(samples[i]));
    samples[i][0] = (void *) uc->uc_mcontext.gregs[REG_RIP];
    // buf[0] = handler, buf[1] = sigreturn trampoline, buf[2] = interrupted frame's caller (usually)
    for (int k = 2, j = 1; k < n && j < DEPTH; k++, j++) samples[i][j] = buf[k];
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    stamps[i] = ts.tv_sec + ts.tv_nsec * 1e-9;
    nsamp = i + 1;
}

static void dump(void) {
    if (!active) return;
    active = 0;
    timer_delete(tm);
    const char * out = getenv("PMP_OUT");
    char path[512];
    snprintf(path, sizeof path, "%s.%d", out ? out : "/tmp/pmp.out", (int) getpid());
    if (nsamp < 100) return; // short-lived helper processes
    FILE * f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "SAMPLES %d\n", nsamp);
    for (int i = 0; i < nsamp; i++) {
        fprintf(f, "%.6f ", stamps[i]);
        for (int j = 0; j < DEPTH; j++) fprintf(f, "%lx ", (unsigned long) samples[i][j]);
        fprintf(f, "\n");
    }
    fprintf(f, "MAPS\n");
    FILE * m = fopen("/proc/self/maps", "r");
    char line[1024];
    while (m && fgets(line, sizeof line, m)) fputs(line, f);
    if (m) fclose(m);
    fclose(f);
}

static void on_term(int sig) {
    dump();
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void init(void) {
    const char * us_s = getenv("PMP_US");
    long us = us_s ? atol(us_s) : 200;
    void * warm[4];
    backtrace(warm, 4); // load libgcc before the first signal
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGRTMIN + 3, &sa, NULL);
    struct sigevent sev;
    memset(&sev, 0, sizeof sev);
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGRTMIN + 3;
    sev._sigev_un._tid = syscall(SYS_gettid);
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &tm) != 0) return;
    struct itimerspec its;
    its.it_interval.tv_sec = 0;
    its.it_interval.tv_nsec = us * 1000;
    its.it_value = its.it_interval;
    timer_settime(tm, 0, &its, NULL);
    active = 1;
    atexit(dump);
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);
}
