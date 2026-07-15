"""
Exact-diagonalization (full CI) reference for the INTERACTING 2D dot:

    H = sum_i [ -1/2 lap_i + 1/2 r_i^2 ]  +  lam * sum_{i<j} exp(-r_ij^2 / (2 s^2))

N spinless (spin-polarized) fermions, Cartesian oscillator basis, shells nx+ny <= S.
The Gaussian interaction is chosen over bare Coulomb deliberately (RESEARCH_LOG 2026-07-15):
its two-body integrals factorize into x/y and are computed EXACTLY (smooth integrand,
high-order Gauss-Legendre), there is no coalescence cusp (so a smooth NN ansatz can in
principle represent the GS — trap failures are then attributable to optimization, not to
cusp representability), and VMC uses the *identical* operator => internally exact benchmark.

Symmetry used: the Hamiltonian conserves total x-parity and y-parity (sum nx mod 2,
sum ny mod 2), so the determinant basis splits into 4 blocks; all 4 are diagonalized.

Validation built in:
  - lam=0 must reproduce the exact non-interacting energies (5.0 for N=3);
  - N=2 ED is cross-checked against an independent radial-ODE solve of the relative
    problem (odd relative angular momentum m=1 for polarized fermions), which exercises
    the two-body integrals AND the Slater-Condon sign conventions end to end.

Usage:
  python tools/ed_dot.py --N 3 --lam 2.0 --s 1.0 --shells 6
  python tools/ed_dot.py --validate          # N=2 cross-check + lam=0 checks
"""
import argparse
import itertools
import math

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


# ---------- 1D harmonic-oscillator orbitals and exact pair-interaction table ----------

def ho_orbitals_1d(nmax, x):
    """psi_n(x), n=0..nmax, orthonormal. Shape (nmax+1, len(x))."""
    psi = np.zeros((nmax + 1, len(x)))
    h_prev = np.ones_like(x)                     # H_0
    psi[0] = np.pi ** -0.25 * np.exp(-x**2 / 2)
    if nmax >= 1:
        h_curr = 2 * x                           # H_1
        psi[1] = np.pi ** -0.25 / np.sqrt(2.0) * h_curr * np.exp(-x**2 / 2)
    for n in range(2, nmax + 1):
        h_prev, h_curr = h_curr, 2 * x * h_curr - 2 * (n - 1) * h_prev
        norm = np.pi ** -0.25 / np.sqrt(2.0 ** n * float(math.factorial(n)))
        psi[n] = norm * h_curr * np.exp(-x**2 / 2)
    return psi


def pair_table_1d(nmax, s, L=9.0, q=300):
    """
    G[n1, n3, n2, n4] = <n1 n2| exp(-(x1-x2)^2/(2 s^2)) |n3 n4>
                      = int int psi_n1(x1) psi_n3(x1) g(x1-x2) psi_n2(x2) psi_n4(x2)
    Gauss-Legendre on [-L, L]^2; integrand is smooth and Gaussian-decaying -> exact
    to machine precision at q=300 for nmax <= 10.
    Also returns the orbital overlap matrix as a quadrature sanity check.
    """
    t, w = np.polynomial.legendre.leggauss(q)
    x = L * t
    wx = L * w
    psi = ho_orbitals_1d(nmax, x)                             # (n+1, q)
    A = np.einsum("ai,ci->aci", psi, psi)                     # (n+1, n+1, q)
    K = np.exp(-np.subtract.outer(x, x) ** 2 / (2 * s * s))   # (q, q)
    Kw = (wx[:, None] * K) * wx[None, :]
    G = np.einsum("aci,ij,bdj->abcd", A, Kw, A)               # [n1,n2,n3,n4] then reorder
    # reorder to G[n1, n3, n2, n4] for the factorized 2D lookup below
    G = np.transpose(G, (0, 2, 1, 3))                         # -> [n1, n3, n2, n4]
    ovl = np.einsum("aci,i->ac", A, wx)                       # should be identity
    return G, np.max(np.abs(ovl - np.eye(nmax + 1)))


# ---------- 2D orbitals, antisymmetrized elements, Slater-Condon ----------

def orbitals_2d(S):
    """(nx, ny) with nx+ny <= S, sorted by energy then lexicographically."""
    orbs = [(nx, ny) for nx in range(S + 1) for ny in range(S + 1) if nx + ny <= S]
    orbs.sort(key=lambda o: (o[0] + o[1], o))
    return orbs


class DotED:
    def __init__(self, N, lam, s, S, q=300):
        self.N, self.lam, self.S = N, lam, S
        self.orbs = orbitals_2d(S)
        self.M = len(self.orbs)
        self.eps = np.array([nx + ny + 1.0 for nx, ny in self.orbs])
        G, ovl_err = pair_table_1d(S, s, q=q)
        assert ovl_err < 1e-10, f"quadrature not converged: overlap error {ovl_err:.2e}"
        nx = np.array([o[0] for o in self.orbs])
        ny = np.array([o[1] for o in self.orbs])
        # V[a,b,c,d] = lam * Gx[ax,cx,bx,dx] * Gy[ay,cy,by,dy]  (<ab|V|cd>, physicist)
        self.V = lam * (G[nx[:, None, None, None], nx[None, None, :, None],
                          nx[None, :, None, None], nx[None, None, None, :]]
                        * G[ny[:, None, None, None], ny[None, None, :, None],
                            ny[None, :, None, None], ny[None, None, None, :]])

    def vas(self, a, b, c, d):
        """antisymmetrized <ab||cd> = <ab|V|cd> - <ab|V|dc>"""
        return self.V[a, b, c, d] - self.V[a, b, d, c]

    def block_dets(self, px, py):
        """all sorted N-tuples with total x-parity px and y-parity py"""
        dets = []
        for det in itertools.combinations(range(self.M), self.N):
            sx = sum(self.orbs[o][0] for o in det) % 2
            sy = sum(self.orbs[o][1] for o in det) % 2
            if (sx, sy) == (px, py):
                dets.append(det)
        return dets

    def build_h(self, dets):
        idx = {d: i for i, d in enumerate(dets)}
        D = len(dets)
        H = lil_matrix((D, D))
        for I, det in enumerate(dets):
            occ = list(det)
            # diagonal
            e = self.eps[occ].sum()
            for i, j in itertools.combinations(occ, 2):
                e += self.vas(i, j, i, j)
            H[I, I] = e
            virt = [o for o in range(self.M) if o not in det]
            # singles i -> a  (one-body part vanishes: h is diagonal in this basis)
            for pi, i in enumerate(occ):
                for a in virt:
                    new = sorted(occ[:pi] + occ[pi + 1:] + [a])
                    J = idx.get(tuple(new))
                    if J is None or J <= I:
                        continue
                    sign = (-1) ** (pi + new.index(a))
                    val = sum(self.vas(i, k, a, k) for k in occ if k != i)
                    if val != 0.0:
                        H[I, J] = sign * val
                        H[J, I] = sign * val
            # doubles (i<j) -> (a<b)
            for (pi, i), (pj, j) in itertools.combinations(enumerate(occ), 2):
                rest = [o for o in occ if o != i and o != j]
                for a, b in itertools.combinations(virt, 2):
                    new = sorted(rest + [a, b])
                    J = idx.get(tuple(new))
                    if J is None or J <= I:
                        continue
                    # phase for (i->a, j->b), i<j, a<b, sorted dets: (-1)^(sum of positions);
                    # verified against the operator algebra for N=2: <{i,j}|H|{a,b}> = +<ij||ab>
                    sign = (-1) ** (pi + pj + new.index(a) + new.index(b))
                    val = self.vas(i, j, a, b)
                    if val != 0.0:
                        H[I, J] = sign * val
                        H[J, I] = sign * val
        return H.tocsr()

    def ground_state(self, n_states=3):
        """diagonalize every parity block; return sorted (E, (px,py)) list"""
        out = []
        for px in (0, 1):
            for py in (0, 1):
                dets = self.block_dets(px, py)
                if not dets:
                    continue
                H = self.build_h(dets)
                k = min(n_states, H.shape[0] - 2)
                if H.shape[0] <= 3:
                    vals = np.linalg.eigvalsh(H.toarray())[:n_states]
                else:
                    vals = eigsh(H, k=k, which="SA", return_eigenvectors=False)
                    vals = np.sort(vals)
                for E in vals:
                    out.append((float(E), (px, py)))
        out.sort()
        return out


# ---------- independent N=2 validation: radial ODE for the relative problem ----------

def n2_relative_ode(lam, s, m=1, umax=25.0, npts=6000):
    """
    N=2, omega=1: E = E_cm + E_rel with E_cm = 1 (2D CM ground state).
    Relative problem (reduced mass 1/2): [-d2/du2 - (1/u) d/du + m^2/u^2 + u^2/4
        + lam exp(-u^2/(2 s^2))] f = E_rel f.
    Polarized fermions -> odd relative parity -> lowest is |m| = 1.
    Substituting f = u^{-1/2} phi removes the first derivative:
        [-d2/du2 + (m^2 - 1/4)/u^2 + u^2/4 + lam g(u)] phi = E_rel phi.
    Finite differences + dense eigh on [h, umax].
    """
    u = np.linspace(0, umax, npts + 1)[1:]
    h = u[1] - u[0]
    diag = 2.0 / h**2 + (m * m - 0.25) / u**2 + u**2 / 4.0 + lam * np.exp(-u**2 / (2 * s * s))
    off = -np.ones(npts - 1) / h**2
    from scipy.linalg import eigh_tridiagonal
    vals = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0]
    return 1.0 + float(vals[0])


def validate(lam, s):
    print("== validation ==")
    # lam = 0, N=3: exact 5.0
    ed0 = DotED(3, 0.0, s, S=5)
    E0 = ed0.ground_state()[0]
    print(f"N=3 lam=0 (S=5):  E = {E0[0]:.10f}  block {E0[1]}   (exact 5.0)")
    assert abs(E0[0] - 5.0) < 1e-9
    # N=2 at coupling: ED vs radial ODE, S-convergence
    ode = n2_relative_ode(lam, s)
    for S in (6, 8, 10):
        ed = DotED(2, lam, s, S=S)
        E = ed.ground_state()[0][0]
        print(f"N=2 lam={lam} s={s} (S={S:2d}): ED = {E:.8f}   ODE = {ode:.8f}   "
              f"diff = {E - ode:+.2e}")
    print("== validation done ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--s", type=float, default=1.0)
    ap.add_argument("--shells", type=int, default=6)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.validate:
        validate(args.lam, args.s)
    else:
        ed = DotED(args.N, args.lam, args.s, args.shells)
        levels = ed.ground_state()
        print(f"N={args.N} lam={args.lam} s={args.s} shells<={args.shells} "
              f"(M={ed.M} orbitals)")
        for E, blk in levels[:8]:
            print(f"  E = {E:.8f}   parity block {blk}")
