"""
Analytic (no VMC needed) check for the L_z-projection trick: for a closed-shell
count N, the holomorphic-trap family sits at L_z = N(N-1)/2 + d, d=0,1,2,...
Projection onto L_z == 0 (mod K) keeps only d with (N(N-1)/2 + d) % K == 0; the
"margin" is the smallest such d (0 = projection totally fails, trap untouched).

Finding (RESEARCH_LOG 2026-07-16): for a FIXED K, margin does not shrink as N
grows (no scaling law K(N) needed) -- but a single fixed K is not safe for all N
(e.g. K=6 gives margin 0 at N=21,28,36,45,...). K must be chosen per N.
"""
import argparse


def margin(N: int, K: int) -> int:
    """Smallest d>=0 such that L_z = N(N-1)/2 + d survives the mod-K projection."""
    lz0 = N * (N - 1) // 2
    return (-lz0) % K


def choose_K(N: int, margin_target: int = 3, K_max: int = 200) -> int:
    """Smallest K achieving at least margin_target; deterministic, O(K_max) to compute."""
    for K in range(2, K_max):
        if margin(N, K) >= margin_target:
            return K
    raise ValueError(f"no K < {K_max} achieves margin >= {margin_target} for N={N}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, help="check margin for this specific K")
    ap.add_argument("--margin-target", type=int, default=3)
    args = ap.parse_args()

    if args.K is not None:
        print(f"N={args.N} K={args.K}: margin d0 = {margin(args.N, args.K)}")
    else:
        K = choose_K(args.N, args.margin_target)
        print(f"N={args.N}: smallest K achieving margin>={args.margin_target} is K={K} "
              f"(margin={margin(args.N, K)})")
