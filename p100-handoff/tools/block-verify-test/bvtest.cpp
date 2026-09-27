// Distribution test of the real common_spec_block_verify (linked from libllama-common):
// random target/draft trees over a small vocab, drafts drawn from q, full speculative loop until
// L tokens are produced; the empirical distribution of the first L tokens must match the target's.
// Includes zero-probability entries (like top-k truncation) in both p and q.
#include "sampling.h"
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

static int V, G, L;
static std::mt19937 tree_rng(1);
static std::map<std::vector<int>, std::vector<double>> Pt, Qt;

static std::vector<double> rnd(bool truncate) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> w(V);
    double s = 0;
    for (auto & x : w) { x = std::pow(u(tree_rng), 3); }
    if (truncate) { w[tree_rng() % V] = 0.0; } // a token outside the support
    for (auto x : w) s += x;
    for (auto & x : w) x /= s;
    return w;
}
static const std::vector<double> & dist(std::map<std::vector<int>, std::vector<double>> & m, const std::vector<int> & pre, bool trunc) {
    auto it = m.find(pre);
    if (it == m.end()) it = m.emplace(pre, rnd(trunc)).first;
    return it->second;
}
static std::vector<llama_token_data> as_td(const std::vector<double> & d) {
    std::vector<llama_token_data> r;
    for (int i = 0; i < V; ++i) if (d[i] > 0) r.push_back({i, 0.0f, (float) d[i]});
    return r;
}
static int samp(const std::vector<double> & d, std::mt19937 & r) {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(r), a = 0;
    for (int i = 0; i < V; ++i) { a += d[i]; if (u < a) return i; }
    for (int i = V - 1; i >= 0; --i) if (d[i] > 0) return i;
    return 0;
}

int main(int argc, char ** argv) {
    V = atoi(argv[1]); G = atoi(argv[2]); L = atoi(argv[3]);
    const long N = atol(argv[4]);
    const bool bug = argc > 5; // sanity: drop the correction token's residual (use p directly)
    std::mt19937 r(7);
    std::map<std::vector<int>, long> cnt;
    std::vector<double> keep;
    for (long n = 0; n < N; ++n) {
        std::vector<int> seq;
        while ((int) seq.size() < L) {
            llama_tokens draft; std::vector<std::vector<llama_token_data>> dists, P;
            std::vector<llama_token> ids;
            std::vector<int> pre = seq;
            for (int i = 0; i < G; ++i) {
                const auto & q = dist(Qt, pre, true);
                // the float the real code sees
                auto qd = as_td(q); std::vector<double> qf(V, 0.0); for (auto & c : qd) qf[c.id] = c.p;
                int x = samp(qf, r);
                dists.push_back(qd); draft.push_back(x); pre.push_back(x);
            }
            pre = seq;
            for (int i = 0; i <= G; ++i) {
                auto pd = as_td(dist(Pt, pre, true)); std::vector<double> pf(V, 0.0); for (auto & c : pd) pf[c.id] = c.p;
                P.push_back(pd); ids.push_back(samp(pf, r));
                if (i < G) pre.push_back(draft[i]);
            }
            llama_token y;
            size_t tau = common_spec_block_verify(P, dists, draft, ids, r, y, keep);
            if (bug && tau < (size_t) G) y = ids[tau];
            for (size_t i = 0; i < tau; ++i) seq.push_back(draft[i]);
            seq.push_back(y);
        }
        seq.resize(L);
        cnt[seq]++;
    }
    // exact target probability of every length-L sequence (with the float-rounded p the code sees)
    double worst = 0, total = 0;
    std::vector<int> s(L, 0);
    for (long k = 0; k < (long) std::pow(V, L); ++k) {
        long t = k; for (int i = 0; i < L; ++i) { s[i] = t % V; t /= V; }
        double pe = 1; std::vector<int> pre;
        for (int i = 0; i < L; ++i) { auto pd = as_td(dist(Pt, pre, true)); double pi = 0; for (auto & c : pd) if (c.id == s[i]) pi = c.p; double z = 0; for (auto & c : pd) z += c.p; pe *= pi / z; pre.push_back(s[i]); }
        total += pe;
        const double emp = (double) cnt[s] / N, se = std::sqrt(std::max(pe * (1 - pe), 1e-12) / N);
        if (pe > 0 || cnt[s] > 0) worst = std::max(worst, std::fabs(emp - pe) / se);
    }
    printf("V=%d G=%d L=%d N=%ld%s: worst |empirical - exact| = %.2f standard errors over %d sequences\n",
           V, G, L, N, bug ? " (planted bug)" : "", worst, (int) std::pow(V, L));
    return 0;
}
