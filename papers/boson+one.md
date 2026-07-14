A minimal and universal representation of fermionic wavefunctions
(fermions = bosons + one)
Liang Fu1
1Department of Physics, Massachusetts Institute of Technology, Cambridge, MA 02139, USA
Representing fermionic wavefunctions efficiently is a central problem in quantum physics, chem-
istry and materials science. In this work, we introduce a universal and exact representation of
continuous antisymmetric functions by lifting them to continuous symmetric functions defined on
an enlarged space. Building on this lifting, we obtain a parity-graded representation of fermionic
wavefunctions, expressed in terms of symmetric feature variables that encode particle configuration
and antisymmetric feature variables that encode exchange statistics. This representation is both
exact and minimal: the number of required features scales as D∼ N d (d is spatial dimension) or
D∼ N depending on the symmetric feature maps employed. Our results provide a rigorous mathe-
matical foundation for efficient representations of fermionic wavefunctions and enable scalable and
systematically improvable neural network solvers for many-electron systems.
A fundamental principle of quantum mechanics is that
identical particles are indistinguishable. Consequently,
exchanging two identical particles can change the many-
body wavefunction only by an overall phase factor: +1
for bosons (symmetric) and−1 for fermions (antisym-
metric). The antisymmetry of fermionic wavefunctions
embodies the Pauli exclusion principle, which underlies
the stability of matter [1] and governs the electronic prop-
erties of metals and insulators [2].
Despite its central importance in physics, chemistry
and materials science, efficiently representing fermion
states remains a major challenge. In essence, the prob-
lem is this: how can one efficiently express an arbitrary
N-fermion wavefunction ψ(r1,...,rN), which satisfies the
antisymmetric condition
ψ(r1,...,rN) = (−1)σψ(rσ(1),...,rσ(N)), σ∈SN (1)
in terms of unconstrained functions? The standard
approach is to expand ψ in terms of Slater deter-
minants constructed from a single-particle basis set
ϕ1(r),...,ϕL(r). This representation becomes exact only
in the complete basis set limit (L →∞). Even when
truncated to a finite basis with L∝N, the number of re-
quired Slater determinants still scales exponentially with
particle number N. Alternatively, one may construct ψ
by explicitly antisymmetrizing a general function over all
particle permutations, but this approach is also inefficient
due to the N! permutations involved.
Historically, a class of fermionic wavefunctions was
constructed by multiplying a Slater determinant by a
Jastrow factor [3] and incorporates backflow transfor-
mations of particle coordinates [4, 5]. More recently,
neural-network-based fermionic wavefunctions [6–8]—
most prominently the FermiNet [9]—have extended this
backflow-determinant framework, achieving greatly en-
hanced expressive power and impressive accuracy.
Although a sufficiently large neural network can ap-
proximate any continuous function over a compact do-
main arbitrarily well (the universal approximation the-
orem [10]), it remains unclear whether continuous anti-
symmetric functions can be universally represented by
existing fermionic neural networks—which enforces an-
tisymmetry by design. Previous arguments [9, 11] sug-
gesting that a single FermiNet determinant can represent
any antisymmetric function rely on sorting the particle
coordinates—a procedure that necessarily introduces dis-
continuities in spatial dimensions d>1, and is therefore
beyond the reach of neural network approximators.
Recently, in an important and stimulating mathemat-
ical work [12], Chen and Lu showed that any continuous
antisymmetric function can be expressed as a composi-
tion of an odd function with a fixed set of antisymmetric
basis functions—or feature functions in the language of
machine learning. In contrast to the Slater determinant
basis, the required number of such functions, Da, scales
only polynomially with particle number Da ∼Nd+2 for
d>1 and N ≫d. This result is the first efficient repre-
sentation of fermionic wavefunctions.
In this work, we introduce a new approach to exactly
represent antisymmetric functions and provide a new uni-
versal representation. We show that any continuous an-
tisymmetric function ψ(r1,...,rN) can be lifted to a con-
tinuous function Ψ(r1,....,rN,η) by embedding particle
coordinates into an enlarged space that includes auxil-
iary coordinates η. The lifted function Ψ recovers ψ
when restricted to the physical sector η= η(r1,...,rN)
as specified by an antisymmetric mapping from particle
coordinates to ηcoordinates, and Ψ is symmetric in par-
ticle coordinates r1,...,rN. Equivalently, this construc-
tion provides an exact transformation from fermions to
bosons through the introduction of just one ancilla par-
ticle that carries the “signature” of Fermi statistics.
Building on this lifting, we obtain a universal rep-
resentation: any continuous antisymmetric function
ψ(r1,...,rN) can be expressed as a continuous function
f(ξ,η) of “generalized coordinates”—namely, symmet-ric feature variables ξthat encode particle configuration
in quotient space (Rd)N/SN and antisymmetric feature
variables η that encode Fermi statistics. This “parity-
graded” representation guarantees fermionic antisymme-
try without requiring any explicit constraint on f. It is
both exact and minimal. The feature dimension scales
with particle number as D∼Nd when a standard set of
symmetric features is employed. Using a random selec-
tion of symmetric features [13], the feature dimension in
our representation can be further reduced to D∼N for
N ≫d in any spatial dimension.
The key component of our representation is the anti-
symmetric feature function η(r1,....,rN)—the encoder of
Fermi statistics that links ψand Ψ. We identify canonical
choices of ηthat are independent of the target wavefunc-
tion ψ and provide their explicit forms. For d= 2, ηcan
be realized by certain quantum Hall wavefunctions in the
lowest Landau level.
FERMION-BOSON TRANSFORMATION
Our basic idea is simple. Given any continuous anti-
symmetric function ψ(r1,...,rN), we construct a lifted
function Ψ(r1,...,rN,η) defined on an enlarged space
that includes an auxiliary variable η. On the “physi-
cal sector”, specified by a mapping η= η(r1,...,rN),
the lifted function reduces to the original one:
ψ(r1,...,rN) = Ψ (r1,...,rN,η(r1,...,rN)). (2)
We show that for a suitably chosen mapping η, the lifted
function Ψ admits a much simpler representation, from
which an efficient and universal representation of anti-
symmetric functions follows.
Let ψ: Ω →C be a continuous antisymmetric function
of R= (r1,...,rN), defined on a compact domain Ω ⊂
(Rd)N. Let η : Ω →Rc be a feature map that maps
each physical coordinate to cauxiliary variables, η(R) =
(η1(R),...,ηc(R)). Our construction uses a suitably-
chosen feature map that satisfies the following required
properties.
Property 1. The feature map is continuous and an-
tisymmetric: for any a = 1,...,c and any permutation
σ∈SN,
ηa(r1,...,rN) = (−1)σηa(rσ(1),...,rσ(N)). (3)
In particular, Property 1 implies that η(R) = 0 (the
zero vector in Rc) whenever Rcontains a collision, i.e.,
ri = rj for some i̸= j.
Property 2. Conversely, η(R) = 0 only if Rcontains
a collision.
A feature map ηthat satisfies Properties 1 and 2 is able
to (1) exactly detect every collision-free configuration;
2
and (2) distinguish even- and odd-permutations of any
collision-free configuration through opposite signs of η.
We refer to such a map as a signature encoder.
Assuming a signature encoder exists, we define a lifted
function on the union of two opposite embeddings in the
enlarged space, (R,η1) with η1 = η(R) and (R,η2) with
η2 =−η(R), as follows:
Ψ(R,η1) = ψ(R), Ψ(R,η2) =−ψ(R). (4)
Importantly, this definition is well defined. By Property
1, for any collision-free configuration, we have η1 ̸= η2,
so the two values Ψ(R,η1) and Ψ(R,η2) can be assigned
separately according to Eq.(4). For any collision con-
figuration, we have η1 = η2 = 0, and simultaneously
ψ(R) = 0 due to antisymmetry, which remains consistent
with Eq.(4). Thus, the lifted function Ψ realizes ψ as a
two-sheeted cover of configuration space distinguished by
opposite embeddings ±η(R).
By its definition, the lifted function has the following
properties:
Property 3. Ψ is odd in the auxiliary variable η:
Ψ(r1,...,rN,−η) =−Ψ(r1,...,rN,η). (5)
and
Property 4. Ψ is symmetric under particle permuta-
tion in R—that is, for any permutation σ,
Ψ(r1,...,rN,η) = Ψ(rσ(1),...,rσ(N),η). (6)
This follows because both ψ and ηacquire the same sign
factor (−1)σ under permutations due to antisymmetry.
Since Ψ is odd in η (Property 3), these sign changes
cancel, leaving the lifted function invariant under particle
permutations in the enlarged space.
By continuity, the lifted function, initially defined on
the two opposite embeddings, can be extended to a con-
tinuous function on a larger region of the (R,η)-space
(Tietze extension theorem). Conversely, for any contin-
uous function Ψ(R,η) in the enlarged space that sat-
isfies Eqs. (5) and (6), restricting to the “physical sec-
tor” defined by the signature encoder—that is, setting
η= η(R)—always yields an antisymmetric function.
Therefore, by the above procedure employing a sig-
nature encoder, any continuous antisymmetric function
ψ(R) can be lifted to a continuous function Ψ(R,η) that
is invariant under the permutation of particle coordi-
nates R and odd in auxiliary coordinate η, and can be
recovered from it. Thus, the task of representing ψ can
be achieved by finding a universal representation of Ψ in
the enlarged space. This constitutes our first main result.
Remark. The lifted function Ψ(r1,...,rN,η1,...,ηc)
can be interpreted as the wavefunction of a system of N
bosons in d dimensions, augmented by a single ancillaparticle in cdimensions. The ancilla particle is restricted
to odd-parity modes in ηspace and is entangled with the
N bosons. Remarkably, any state of N fermions can be
retrieved from the corresponding state of this system. In
this sense, the bosonic wavefunction Ψ captures fermionic
antisymmetry of ψ through the addition of just one an-
cilla particle.
SIGNATURE ENCODER
The signature encoder η(R) plays the essential role
in recovering the fermionic wavefunction ψ(R) from the
˜
bosonic wavefunction
Ψ(R,η); equivalently, it imple-
ments the boson-fermion transformation. In this section,
we present the explicit form of η(R).
By Property 1, η(R) = (η1(R),...,ηc(R)) ∈Rc may be
viewed as a collection of real-valued fermionic functions,
each ηa changing sign under odd permutations. Property
2 requires that these functions vanish simultaneously only
on collision configurations. This requirement is nontriv-
ial: by continuity, the zero set of a single real-valued
fermionic function typically forms a codimension-1 sub-
manifold of configuration space, whereas the collision set
{ri = rj,i ̸= j}has codimension d. Thus, for d > 1,
a single fermionic function generally has far more zeros
than the collision locus mandated by the Pauli principle.
The natural question, then, is whether one can choose
a finite family η1,...,ηc so that at every collision-
free configuration at least one component is nonzero—
equivalently, the common zero set {η= 0} coincides
with the collision set—and, if so, what is the minimal
dimension c required?
One and Two Dimensions
We now present a minimal set of antisymmetric real
functions that serve as signature encoders in one and two
dimensions.
d= 1 : η=
(xi−xj) (7)
i<j
d= 2 : η1 + iη2 =
(zi−zj) (8)
i<j
where zi = xi + iyi are complex coordinates.
For d= 1, a single function η—the real Vandermonde
polynomial of x1,...,xN—suffices: as a product of all
pairwise differences xi−xj, it is antisymmetric and van-
ishes only on the collision set {xi = xj,i ̸= j}, thus
satisfying Property 1 and 2.
For d = 2, two antisymmetric real functions η1,η2 of
the coordinates (x1,y1),...,(xN,yN) are required. They
3
correspond to the real and imaginary parts of a complex
Vandermonde polynomial of the complex coordinates, as
given in Eq. (8). As a product of zi−zj, the pair (η1,η2)
vanishes simultaneously only if zi = zj or equivalently
(xi,yi) = (xj,yj) for some i̸= j.
Eqs.(9) and (8) can be generalized into pair product
wavefunctions as follows:
d= 1 : η=
∆(xi−xj) (9)
i<j
d= 2 : η1 + iη2 =
∆(ri−rj), (10)
i<j
with ∆ any odd function that vanishes only at the origin.
As a further generalization, η may be multiplied by a
symmetric product factor iρ(ri) with ρ(r) any function
that is everywhere nonzero.
As an interesting example in d= 2, a signature encoder
is realized by quantum Hall wavefunctions in the lowest
Landau level,
η1 + iη2 =
(zi−zj)q ×
exp −|zi|2
, (11)
i<j
i
where q is an odd positive integer. In particular, q = 1
corresponds to the ν = 1 quantum Hall state—a com-
pletely full Landau level—whose wavefunction equals a
Slater determinant det[ϕi(zj)]] with single-particle or-
bitals ϕi(z) = zi−1 exp −|z|2 . On the other hand, q= 3
corresponds to the ν =
1
3 Laughlin wavefunction, which
cannot be written as a single Slater determinant. Both
serve as valid signature encoders capable of transforming
any fermionic wavefunction ψ(r1,...,rN) into a corre-
sponding bosonic wavefunction Ψ(r1,...,rN,η), regard-
less of whether ψ resides within the lowest Landau level.
All Dimensions
To construct ηfunctions in higher dimensions d>2, it
is tempting to consider generalizations of Vandermonde
determinants from the real or complex coordinate to the
quaternion coordinate, such as q= x+ iy+ jz+ kw in
d = 4. Because the quaternion ring has no zero divisor,
the quaternion Vandermonde polynomial i<j(qi−qj)
vanishes only at collision, thereby satisfying Property
2. However, since quaternions are noncommutative, the
quaternion Vandermonde polynomial is generally not an-
tisymmetric under particle permutation, and thus fails to
satisfy Property 1.
To proceed, we consider a slight variant of
the problem—representing antisymmetric wavefunctions
subject to periodic boundary condition, which describe
fermions on a torus Td rather than on a compact domain
in Rd as considered so far. Mathematically, ψ satisfies
ψ(...,ri,...) = ψ(...,ri + L,...) (12)for any L=
d
l=1 nlLl with (n1,...,nd) ∈ Zd, where
L1,...,Ld form a set of linearly independent supercell
vectors.
Such a periodic function can be expressed in terms of
its Fourier expansion,
ψ(r1,...,rN) =
eiki·ri
˜
ψ(k1,...,kN) (13)
k1 ,...,kN
where the allowed wavevectors ki satisfy eiki·Ll = 1 for
all i,land thus form a lattice in reciprocal space. Specif-
ically, each wavector can be written as
ki =
d
l=1
mlgl, (m1,...,md) ∈Zd (14)
where g1,...,gd are a complete set of reciprocal supercell
vectors defined by gk·Ll = 2πδkl, k,l= 1,...,d.
Complementary to ψ(r1,...,rN) in continuous real
˜
space,
ψ(k1,...,kN) represents the probability ampli-
tude for finding N fermions occupying the set of plane
wave modes at discrete wavevectors k1,...,kN. Related
˜
through the Fourier transform, ψ and
ψ describe the
same quantum state of N fermions in first quantiza-
tion, and both must be antisymmetric under particle ex-
change.
The fermion-boson transformation introduced in Sec-
˜
tion 1 applies equally well to
ψ. We now introduce a
single function η(K) as a signature encoder for antisym-
˜
metric functions
ψ(K), where K= (k1,...,kN) ∈(Zd)N
,
valid in arbitrary dimension.
η(K) =
i<j
δ(ki−kj) (15)
with δ(k) any odd function over the reciprocal lattice
that vanishes only at the origin. A simple choice is
δ(k) = w·k, where wis a fixed vector defining an incom-
mensurate direction in reciprocal space, i.e., w·k = 0
implies k = 0 for k in reciprocal lattice. Equivalently,
the hyperplane w·k= 0 intersects the reciprocal lattice
only at the origin, ensuring that δ(k) = 0 if and only if
k= 0. Such incommensurate directions are generic: for
almost any choice of w, this condition is satisfied. Hence,
the function η(K) satisfies Properties 1 and 2 and thus
serves as a valid signature encoder.
To summarize this section, we provide the explicit
forms of the signature encoder η for representing
fermionic wavefunctions in coordinate space, defined ei-
ther on a compact domain or under periodic boundary
conditions. For a compact domain, a convenient choice of
signature encoder is Vandermonde polynomial of the real
coordinates for d= 1, or of the complex coordinates for
d= 2. For periodic boundary conditions, a canonical sig-
nature encoder can be defined for the dual wavefunction
4
over the reciprocal lattice, given by the Vandermonde
polynomial of the one-dimensional projection of the re-
ciprocal lattice coordinates, valid in arbitrary dimension.
Together with Section 1, this completes the exact
transformation from fermions to bosons through the in-
troduction of a single ancilla particle in the η space,
achieved by Eq.(2), with the mapping from physical space
to η coordinate defined by the signature encoders pre-
sented above. As demonstrated in all the cases, the di-
mensionality of the ηspace is minimal: c= 1 or 2.
UNIVERSAL REPRESENTATION
Finally, we turn to the universal representation of lifted
functions Ψ on (R,η)-space that satisfy Properties 3
and 4—namely, permutation invariance in r1,...,rN and
oddness in η. The same approach also applies to lifted
functions on (K,η)-space.
Observe that any such Ψ can be expressed as an odd
superposition
Ψ(R,η) = Ψ0(R,η)−Ψ0(R,−η)
2 , (16)
where Ψ0 is invariant under permutations in r1,...,rN
and unconstrained in its dependence on η.
Thus, the task reduces to representing Ψ0(R,η)—a
family of continuous symmetric functions parameterized
by η. This can be achieved naturally by generalizing
the representation of symmetric functions. Following
the Deep Sets framework [14], universal representation
of symmetric functions has been extensively studied, and
efficient constructions are now well established [15–17].
The key idea is that any continuous symmetric func-
tion ϕ(R) can be represented by the composition of
a permutation-invariant continuous map from particle
coordinate R to a m-dimensional feature vector ξ=
(ξ1,...,ξm) ∈Rm, followed by a continuous function f
defined on the feature space:
ϕ(R) = f(ξ(R)). (17)
Importantly, there exists a feature map ξ that uniquely
encodes particle configuration up to permutation, that is,
ξ(r1,...,rN) = ξ(r′
1,...,r′
N) if and only if there exists a
permutation σ such that (r′
1,...,r′
N) = (rσ(1),...,rσ(N)).
We refer to such a mapping ξ as a set embedding, since
it provides an injective embedding of the unordered set
of particle coordinates {ri}(i.e., the configuration space
(Rd)N/SN) into the feature space. Consequently, any
ϕ(R) can be expressed as a continuous function of the
feature vector, defined as f(ξ) = ϕ(R= ξ−1).
Specifically, for symmetric functions on a compact do-
main, a set embedding can be realized through a set ofrk1
kd
i,1...r
i,d, 1 ≤k1 +...+ kd ≤N. (18)
multisymmetric power sums (which are manifestly per-
mutation invariant):
N
i=1
In physics terms, these power sums correspond to multi-
pole moments of particles. It is know that the coordinates
r1,...,rN can be uniquely recovered up to permutation
from the values of these power sums [18–20]. The number
of these power sums, which determines the dimensional-
ity of the symmetric feature space m, is m=
N+d
d−1,
which scales as m ∼Nd for N ≫d. Alternatively, a
recent work [13] shows that permutation-invariant em-
beddings can be achieved by randomly selecting 2dN+ 1
symmetric functions, thus reducing the feature space di-
mension to O(N) for any d.
For our problem, Ψ0(R,η) is a symmetric function of
R parameterized by η. Thus, it can be represented by
generalizing Eq.(17) to make f dependent on η. Combin-
ing the result with Eqs. (2) and (16), we establish that
any continuous antisymmetric function ψ can be exactly
represented—through the signature encoder η and the
set embedding ξ—as a continuous function f of the fea-
ture space of (ξ,η) ∈Rm+c
,
ψ(R) = f(ξ(R),η(R))−f(ξ(R),−η(R))
2 , (19)
This representation is universal. It guarantees the an-
tisymmetry of ψwithout imposing any explicit constraint
on f, but rather through the combination effect of (1) the
permutation-invariant set embedding ξ, (2) the antisym-
metric signature encoder η, and (3) oddness in η intro-
duced by antisymmetrization. Since this representation
uses both symmetric (ξ) and antisymmetric (η) features
of particle coordinates R, we refer to it as a “parity-
graded representation” of fermionic wavefunctions. This
constitutes our second main result.
Since our construction only uses a minimal number
(c = 1,2) of antisymmetric features, the feature space
dimension in our (ξ,η)-representation, D= m+c, scales
with the particle number N in the same way as the fea-
ture space scaling for symmetric functions—namely, D∼
Nd or D∼N depending on the set embedding method
employed. For comparison, the universal representation
of Ref.[12] only employs antisymmetric features, and thus
requires a significantly larger feature space dimension
Da ∼ Nd+2 in d > 1. Hence, our parity-graded rep-
resentation constitutes the most efficient method for rep-
resenting general antisymmetric functions.
As an example illustrating how the (ξ,η)-
representation works, let us consider repre-
senting fractional quantum Hall wavefunctions
such as the ν = 1/3 Laughlin wavefunction
5
ψ1
3
= i<j(zi−zj)3 exp−
i|zi|2/2 , using the signa-
ture map η1 +iη2 = i<j(zi−zj) exp−
i|zi|2/2 . One
can write ψ1
3 = (η1 +iη2)×P, where P= i<j(zi−zj)2
is a symmetric polynomial of complex coordinates
z1,...,zN. Remarkably, P can be neatly expressed
in terms of power sums ξk = izk
i through Hankel
determinant:
P = det[ξi+j−2]1≤i,j≤N. (20)
Therefore, the (ξ,η)-representation of ν = 1/3 Laughlin
wavefunction takes an analytical form:
f(ξ,η) = (η1 + iη2) det[ξi+j−2], (21)
using a total of 2N feature variables (discounting ξ0 =
N): ξ1,...,ξ2N−2,η1,η2.
DISCUSSION
We emphasize that the function f representing a con-
tinuous antisymmetric function ψ is itself guaranteed
to be continuous. This follows from the continuity-
preserving nature of each step in the construction, ψ →
Ψ →Ψ0 →f, and from the use of continuous feature
maps ξand η.
One may wonder whether the universal representation
Eq.(19) could be further simplified. A seemingly plausi-
ble approach is to express ψ(R) as a product of a signa-
ture map η(R) and a symmetric function ϕ(R), where
ϕ(R) is defined as the ratio ψ(R)/η(R). Since η(R) is
nonzero away from collisions, ϕ(R) appears well behaved
in that region. However, there is no guarantee that ϕ(R)
remains continuous in the vicinity of collisions. Consider
this example in d= 2: ψ = ¯ z1−
¯
z2 and η= z1−z2; the
ratio ψ/η depends on the direction in which z1−z2 ap-
proaches 0, and thus fails to be continuous near z1 = z2.
While Eq. (19) provides an exact and universal rep-
resentation of continuous antisymmetric functions, the
continuous function on feature space f(ξ,η) may not in-
herit the same regularity as the target function ψ(R) in
coordinate space. To illustrate this point, consider rep-
resenting an antisymmetric function ψ(x1,x2) = x1−x2
using the signature encoder η= (x1−x2)3 and the set em-
bedding ξ= (x1 +x2,x2
1 +x2
2) which uniquely determines
the unordered pair {x1,x2}. Let E denote the image of
the feature map (x1,x2) →(ξ1,ξ2,η) and define a region
Σ of feature space surrounding E defined by |η|≤2D3/2
and D ≥0, where D = 2ξ2−ξ2
1 . A function f over Σ
can then be defined as f(ξ1,ξ2,η) = η/D for D>0 and
0 on D= 0. Importantly, f defined in this way recovers
ψ when restricted to E, and is continuous throughout Σ.
Although it provides an exact representation of ψ, f is
not differentiable at points with D= 0.On the other hand, the universal approximation
theorem guarantees that any continuous function f—
differentiable or not—can be approximated to arbitrary
accuracy by a sufficiently large neural network. In the
above example, the function f= η/D is well approxi-
mated by fϵ = ηD/(D2 + ϵ) with a small ϵ>0, which is
smooth everywhere in R3 and can be efficiently learned
by a neural network.
Therefore, the (ξ,η)-representation provides a natural
foundation for constructing universal approximators for
continuous antisymmetric functions through neural net-
works. In this framework, the antisymmetry of the wave-
function is enforced exactly through the parity-graded
structure of Eq. (19), while the continuous function f
can be implemented as a neural network without any
symmetry constraints.
The set embedding ξ(R), which maps particle coordi-
nates to permutation-invariant features, need not be re-
stricted to analytic forms such as multisymmetric power
sums. In practice, ξcan be implemented as a learnable,
permutation-invariant architecture—such as the Deep
Sets network [14] or its extensions [15, 16, 21]. Simi-
larly, the continuous function f(ξ,η) may be realized by
a deep neural network (for example, a multilayer percep-
tron) capable of capturing complex correlations between
the symmetric (ξ) and antisymmetric (η) features.
This construction offers a principled approach to
fermionic neural network wavefunctions in which the an-
tisymmetry is enforced by design, and the universal rep-
resentational power is limited only by the capacity of
the neural architectures used for f and ξ. In this sense,
our (ξ,η)-representation unifies a rigorous mathematical
foundation with the flexibility of modern deep learning
and opens a direct path to efficient, scalable and sys-
tematically improvable neural network solver for Fermi
systems.
Finally, we note that the (ξ,η) representation nat-
urally incorporates the indistinguishability of identi-
cal particles through the definition of the configuration
space, which is not the Cartesian product of single par-
ticle spaces but rather its quotient space (Rd)N/SN [22].
This configuration space is faithfully captured by the
symmetric ξcoordinates, while its double covering, rep-
resented by the antisymmetric ηcoordinate, gives rise to
Fermi statistics.