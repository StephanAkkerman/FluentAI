# Natural Language Toolkit: ALINE
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Greg Kondrak <gkondrak@ualberta.ca>
#         Geoff Bacon <bacon@berkeley.edu> (Python port)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
ALINE
https://webdocs.cs.ualberta.ca/~kondrak/
Copyright 2002 by Grzegorz Kondrak.

ALINE is an algorithm for aligning phonetic sequences, described in [1].
This module is a port of Kondrak's (2002) ALINE. It provides functions for
phonetic sequence alignment and similarity analysis. These are useful in
historical linguistics, sociolinguistics and synchronic phonology.

ALINE has parameters that can be tuned for desired output. These parameters are:
- C_skip, C_sub, C_exp, C_vwl
- Salience weights
- Segmental features

In this implementation, some parameters have been changed from their default
values as described in [1], in order to replicate published results. All changes
are noted in comments.

Example usage
-------------

# Get optimal alignment of two phonetic sequences

>>> align('θin', 'tenwis') # doctest: +SKIP
[[('θ', 't'), ('i', 'e'), ('n', 'n'), ('-', 'w'), ('-', 'i'), ('-', 's')]]

[1] G. Kondrak. Algorithms for Language Reconstruction. PhD dissertation,
University of Toronto.
"""

from constants import *

try:
    import numpy as np
except ImportError:
    np = None


# === Algorithm ===
def align(str1, str2, epsilon=0):
    """
    Compute the alignment of two phonetic strings.

    :param str str1: First string to be aligned
    :param str str2: Second string to be aligned

    :type epsilon: float (0.0 to 1.0)
    :param epsilon: Adjusts threshold similarity score for near-optimal alignments

    :rtype: list(list(tuple(str, str)))
    :return: Alignment(s) of str1 and str2

    (Kondrak 2002: 51)
    """
    if np is None:
        raise ImportError("You need numpy in order to use the align function")

    assert 0.0 <= epsilon <= 1.0, "Epsilon must be between 0.0 and 1.0."
    m = len(str1)
    n = len(str2)
    # This includes Kondrak's initialization of row 0 and column 0 to all 0s.
    S = np.zeros((m + 1, n + 1), dtype=float)

    # If i <= 1 or j <= 1, don't allow expansions as it doesn't make sense,
    # and breaks array and string indices. Make sure they never get chosen
    # by setting them to -inf.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            edit1 = S[i - 1, j] + sigma_skip(str1[i - 1])
            edit2 = S[i, j - 1] + sigma_skip(str2[j - 1])
            edit3 = S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1])
            if i > 1:
                edit4 = S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2 : i])
            else:
                edit4 = -inf
            if j > 1:
                edit5 = S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2 : j])
            else:
                edit5 = -inf
            S[i, j] = max(edit1, edit2, edit3, edit4, edit5, 0)

    T = (1 - epsilon) * np.amax(S)  # Threshold score for near-optimal alignments

    alignments = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S[i, j] >= T:
                alignments.append(_retrieve(i, j, 0, S, T, str1, str2, []))
    return alignments


def _retrieve(i, j, s, S, T, str1, str2, out):
    """
    Retrieve the path through the similarity matrix S starting at (i, j).

    :rtype: list(tuple(str, str))
    :return: Alignment of str1 and str2
    """
    if S[i, j] == 0:
        return out
    else:
        if j > 1 and S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2 : j]) + s >= T:
            out.insert(0, (str1[i - 1], str2[j - 2 : j]))
            _retrieve(
                i - 1,
                j - 2,
                s + sigma_exp(str1[i - 1], str2[j - 2 : j]),
                S,
                T,
                str1,
                str2,
                out,
            )
        elif (
            i > 1 and S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2 : i]) + s >= T
        ):
            out.insert(0, (str1[i - 2 : i], str2[j - 1]))
            _retrieve(
                i - 2,
                j - 1,
                s + sigma_exp(str2[j - 1], str1[i - 2 : i]),
                S,
                T,
                str1,
                str2,
                out,
            )
        elif S[i, j - 1] + sigma_skip(str2[j - 1]) + s >= T:
            out.insert(0, ("-", str2[j - 1]))
            _retrieve(i, j - 1, s + sigma_skip(str2[j - 1]), S, T, str1, str2, out)
        elif S[i - 1, j] + sigma_skip(str1[i - 1]) + s >= T:
            out.insert(0, (str1[i - 1], "-"))
            _retrieve(i - 1, j, s + sigma_skip(str1[i - 1]), S, T, str1, str2, out)
        elif S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1]) + s >= T:
            out.insert(0, (str1[i - 1], str2[j - 1]))
            _retrieve(
                i - 1,
                j - 1,
                s + sigma_sub(str1[i - 1], str2[j - 1]),
                S,
                T,
                str1,
                str2,
                out,
            )
    return out


def sigma_skip(p):
    """
    Returns score of an indel of P.

    (Kondrak 2002: 54)
    """
    return C_skip


def sigma_sub(p, q):
    """
    Returns score of a substitution of P with Q.

    (Kondrak 2002: 54)
    """
    return C_sub - delta(p, q) - V(p) - V(q)


def sigma_exp(p, q):
    """
    Returns score of an expansion/compression.

    (Kondrak 2002: 54)
    """
    q1 = q[0]
    q2 = q[1]
    return C_exp - delta(p, q1) - delta(p, q2) - V(p) - max(V(q1), V(q2))


def delta(p, q):
    """
    Return weighted sum of difference between P and Q.

    (Kondrak 2002: 54)
    """
    features = R(p, q)
    total = 0
    for f in features:
        total += diff(p, q, f) * salience[f]
    return total


def diff(p, q, f):
    """
    Returns difference between phonetic segments P and Q for feature F.

    (Kondrak 2002: 52, 54)
    """
    p_features, q_features = feature_matrix[p], feature_matrix[q]
    return abs(similarity_matrix[p_features[f]] - similarity_matrix[q_features[f]])


def R(p, q):
    """
    Return relevant features for segment comparison.

    (Kondrak 2002: 54)
    """
    if p in consonants or q in consonants:
        return R_c
    return R_v


def V(p):
    """
    Return vowel weight if P is vowel.

    (Kondrak 2002: 54)
    """
    if p in consonants:
        return 0
    return C_vwl


# === Test ===


def demo():
    """
    A demonstration of the result of aligning phonetic sequences
    used in Kondrak's (2002) dissertation.
    """
    data = [pair.split(",") for pair in cognate_data.split("\n")]
    for pair in data:
        alignment = align(pair[0], pair[1])[0]
        alignment = [f"({a[0]}, {a[1]})" for a in alignment]
        alignment = " ".join(alignment)
        print(f"{pair[0]} ~ {pair[1]} : {alignment}")


cognate_data = """jo,ʒə
tu,ty
nosotros,nu
kjen,ki
ke,kwa
todos,tu
una,ən
dos,dø
tres,trwa
ombre,om
arbol,arbrə
pluma,plym
kabeθa,kap
boka,buʃ
pje,pje
koraθon,kœr
ber,vwar
benir,vənir
deθir,dir
pobre,povrə
ðis,dIzes
ðæt,das
wat,vas
nat,nixt
loŋ,laŋ
mæn,man
fleʃ,flajʃ
bləd,blyt
feðər,fEdər
hær,hAr
ir,Or
aj,awgə
nowz,nAzə
mawθ,munt
təŋ,tsuŋə
fut,fys
nij,knI
hænd,hant
hart,herts
livər,lEbər
ænd,ante
æt,ad
blow,flAre
ir,awris
ijt,edere
fiʃ,piʃkis
flow,fluere
staɾ,stella
ful,plenus
græs,gramen
hart,kordis
horn,korny
aj,ego
nij,genU
məðər,mAter
mawntən,mons
nejm,nomen
njuw,nowus
wən,unus
rawnd,rotundus
sow,suere
sit,sedere
θrij,tres
tuwθ,dentis
θin,tenwis
kinwawa,kenuaʔ
nina,nenah
napewa,napɛw
wapimini,wapemen
namesa,namɛʔs
okimawa,okemaw
ʃiʃipa,seʔsep
ahkohkwa,ahkɛh
pematesiweni,pematesewen
asenja,aʔsɛn"""

if __name__ == "__main__":
    demo()
