# Block verification distribution test

Checks that `common_spec_block_verify` (common/sampling.cpp, the function the server calls)
reproduces the target distribution: random target/draft trees over a small vocab, drafts drawn from
the draft distribution, the speculative loop run until L tokens exist, and the empirical
distribution of those L tokens compared with the exact target probability.

    g++ -O2 -std=c++17 -I common -I include -I ggml/include p100-handoff/tools/block-verify-test/bvtest.cpp \
        -L build-opt/bin -lllama-common -lllama -Wl,-rpath,$PWD/build-opt/bin -o bvtest
    ./bvtest 3 4 4 1000000        # vocab 3, draft length 4, 4 tokens, 1M runs
    ./bvtest 3 4 4 1000000 bug    # planted bug (correction token without the residual): must fail

2026-09-26: 1.85 (G=2) and 2.21 (G=4) standard errors worst case over 27/81 sequences (null
behaviour); the planted bug gives 337.
