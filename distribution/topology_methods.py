# coding: utf8

__author__ = "Frederico Campos Freitas"
__version__ = "1.1.2"
__email__ = "fredcfreitas@gmail.com"


import re
import os
# import sys
# import math
import h5py
import itertools
import multiprocessing
# from multiprocessing import Process
import numpy as np
import mdtraj as md
# import SharedArray as sa


# System definitions:
N = 206389
total = np.arange(1, N + 1)
proteins_large = np.arange(1, 53479 + 1)
guanine = np.arange(53480, 55924 + 1)
proteins_small = np.arange(55925, 91695 + 1)
RNA_18S = np.arange(91696, 129644 + 1)
RNA_25S = np.arange(129645, 200260 + 1)
RNA_58S = np.arange(200261, 203614 + 1)
RNA_5S = np.arange(203615, 206194 + 1)
mRNA = np.arange(206195, N + 1)
l24_modeled_f3j77 = np.arange(37205, 37765 + 1)
RNA_25S_modeled_f3j78 = np.arange(171865, 172180 + 1)
# Inserting the parts corrected by Andrei in the rotated structure
S1_corrected_f3j77 = np.arange(59134, 59245 + 1)
RNA_25S_corrected_f3j77 = np.arange(181790, 182144 + 1)
# Defining the atom groups
all_modeled = np.concatenate((l24_modeled_f3j77, RNA_25S_modeled_f3j78))
all_modified = np.concatenate(
    (all_modeled, S1_corrected_f3j77, RNA_25S_corrected_f3j77)
)
all_corrected = np.concatenate((S1_corrected_f3j77, RNA_25S_corrected_f3j77))
large_total = np.concatenate((proteins_large, RNA_25S, RNA_58S, RNA_5S))
small_total = np.concatenate((guanine, proteins_small, RNA_18S, mRNA))
large_no_modeled = large_total[np.isin(large_total, all_modeled, invert=True)]
large_no_modified = large_total[
    np.isin(large_total, all_modified, invert=True)
]
large_no_corrected = large_total[
    np.isin(large_total, all_corrected, invert=True)
]
small_no_modified = small_total[
    np.isin(small_total, all_modified, invert=True)
]
small_no_corrected = small_total[
    np.isin(small_total, all_corrected, invert=True)
]
total_no_modeled = total[np.isin(total, all_modeled, invert=True)]
total_no_modified = total[np.isin(total, all_modified, invert=True)]
total_no_corrected = total[np.isin(total, all_corrected, invert=True)]

# Defining each chain
protein_L1 = np.arange(1, 1609 + 1)
protein_L2 = np.arange(1610, 3527 + 1)
protein_L3 = np.arange(3528, 6609 + 1)
protein_L4 = np.arange(6610, 9359 + 1)
protein_L5 = np.arange(9360, 11735 + 1)
protein_L6 = np.concatenate(
    (np.arange(11736, 12588 + 1), np.arange(12589, 12975 + 1))
)
protein_L7 = np.arange(12976, 14760 + 1)
protein_L8 = np.arange(14761, 16578 + 1)
protein_L9 = np.arange(16579, 18097 + 1)
protein_L10 = np.concatenate(
    (np.arange(18098, 18928 + 1), np.arange(18929, 19815 + 1))
)
protein_L11 = np.arange(19816, 21169 + 1)
protein_L12 = np.concatenate(
    (
        np.arange(21170, 21241 + 1),
        np.arange(21242, 21363 + 1),
        np.arange(21364, 21872 + 1),
    )
)
protein_L13 = np.arange(21873, 23415 + 1)
protein_L14 = np.arange(23416, 24469 + 1)
protein_L15 = np.arange(24470, 26190 + 1)
protein_L16 = np.arange(26191, 27746 + 1)
protein_L17 = np.arange(27747, 29189 + 1)
protein_L18 = np.arange(29190, 30631 + 1)
protein_L19 = np.arange(30632, 32153 + 1)
protein_L20 = np.arange(32154, 33599 + 1)
protein_L21 = np.arange(33600, 34876 + 1)
protein_L22 = np.arange(34877, 35672 + 1)
protein_L23 = np.arange(35673, 36676 + 1)
protein_L24 = np.arange(36677, 37765 + 1)
protein_L25 = np.arange(37766, 38734 + 1)
protein_L26 = np.arange(38735, 39728 + 1)
protein_L27 = np.arange(39729, 40821 + 1)
protein_L28 = np.arange(40822, 41995 + 1)
protein_L29 = np.arange(41996, 42458 + 1)
protein_L30 = np.arange(42459, 43201 + 1)
protein_L31 = np.arange(43202, 44091 + 1)
protein_L32 = np.arange(44092, 45111 + 1)
protein_L33 = np.arange(45112, 45962 + 1)
protein_L34 = np.arange(45963, 46843 + 1)
protein_L35 = np.arange(46844, 47813 + 1)
protein_L36 = np.arange(47814, 48585 + 1)
protein_L37 = np.arange(48586, 49267 + 1)
protein_L38 = np.arange(49268, 49880 + 1)
protein_L39 = np.arange(49881, 50317 + 1)
protein_L40 = np.arange(50318, 50735 + 1)
protein_L41 = np.arange(50736, 50969 + 1)
protein_L42 = np.arange(50970, 51817 + 1)
protein_L43 = np.arange(51818, 52512 + 1)
protein_P0 = np.concatenate(
    (np.arange(52513, 53359 + 1), np.arange(53360, 53479 + 1))
)
# Starting small subunit
GNBPSBLP = np.arange(53480, 55924 + 1)
protein_S0 = np.arange(55925, 57536 + 1)
protein_S1 = np.arange(57537, 59245 + 1)
protein_S2 = np.arange(59246, 60880 + 1)
protein_S3 = np.arange(60881, 62614 + 1)
protein_S4 = np.arange(62615, 64683 + 1)
protein_S5 = np.arange(64684, 66293 + 1)
protein_S6 = np.arange(66294, 68113 + 1)
protein_S7 = np.arange(68114, 69594 + 1)
protein_S8 = np.concatenate(
    (np.arange(69595, 70571 + 1), np.arange(70572, 71084 + 1))
)
protein_S9 = np.arange(71085, 72578 + 1)
protein_S10 = np.arange(72579, 73395 + 1)
protein_S11 = np.arange(73396, 74640 + 1)
protein_S12 = np.arange(74641, 75575 + 1)
protein_S13 = np.arange(75576, 76768 + 1)
protein_S14 = np.arange(76769, 77710 + 1)
protein_S15 = np.arange(77711, 78701 + 1)
protein_S16 = np.arange(78702, 79807 + 1)
# End of bundle 1 - original PDB files
protein_S17 = np.arange(79808, 80772 + 1)
protein_S18 = np.arange(80773, 81965 + 1)
protein_S19 = np.arange(81966, 83078 + 1)
protein_S20 = np.arange(83079, 83934 + 1)
protein_S21 = np.arange(83935, 84619 + 1)
protein_S22 = np.arange(84620, 85641 + 1)
protein_S23 = np.arange(85642, 86763 + 1)
protein_S24 = np.arange(86764, 87837 + 1)
protein_S25 = np.arange(87838, 88400 + 1)
protein_S26 = np.arange(88401, 89169 + 1)
protein_S27 = np.arange(89170, 89780 + 1)
protein_S28 = np.arange(89781, 90278 + 1)
protein_S29 = np.arange(90279, 90722 + 1)
protein_S30 = np.arange(90723, 91197 + 1)
protein_S31 = np.arange(91198, 91695 + 1)
# End of proteins
ribosomal_RNA_18S = np.arange(91696, 129644 + 1)
# End of bundle 2 - original PDB files
ribosomal_RNA_25S = np.concatenate(
    (
        np.arange(129645, 139292 + 1),
        np.arange(139293, 171354 + 1),
        np.arange(171355, 172453 + 1),
        np.arange(172454, 200260 + 1),
    )
)
ribosomal_RNA_58S = np.arange(200261, 203614 + 1)
ribosomal_RNA_5S = np.arange(203615, 206194 + 1)
messengerRNA = np.arange(206195, 206389 + 1)

ribosome_dic = {}
ribosome_dic["protein_L1"] = np.arange(1, 1609 + 1)
ribosome_dic["protein_L2"] = np.arange(1610, 3527 + 1)
ribosome_dic["protein_L3"] = np.arange(3528, 6609 + 1)
ribosome_dic["protein_L4"] = np.arange(6610, 9359 + 1)
ribosome_dic["protein_L5"] = np.arange(9360, 11735 + 1)
ribosome_dic["protein_L6"] = np.concatenate(
    (np.arange(11736, 12588 + 1), np.arange(12589, 12975 + 1))
)
ribosome_dic["protein_L7"] = np.arange(12976, 14760 + 1)
ribosome_dic["protein_L8"] = np.arange(14761, 16578 + 1)
ribosome_dic["protein_L9"] = np.arange(16579, 18097 + 1)
ribosome_dic["protein_L10"] = np.concatenate(
    (np.arange(18098, 18928 + 1), np.arange(18929, 19815 + 1))
)
ribosome_dic["protein_L11"] = np.arange(19816, 21169 + 1)
ribosome_dic["protein_L12"] = np.concatenate(
    (
        np.arange(21170, 21241 + 1),
        np.arange(21242, 21363 + 1),
        np.arange(21364, 21872 + 1),
    )
)
ribosome_dic["protein_L13"] = np.arange(21873, 23415 + 1)
ribosome_dic["protein_L14"] = np.arange(23416, 24469 + 1)
ribosome_dic["protein_L15"] = np.arange(24470, 26190 + 1)
ribosome_dic["protein_L16"] = np.arange(26191, 27746 + 1)
ribosome_dic["protein_L17"] = np.arange(27747, 29189 + 1)
ribosome_dic["protein_L18"] = np.arange(29190, 30631 + 1)
ribosome_dic["protein_L19"] = np.arange(30632, 32153 + 1)
ribosome_dic["protein_L20"] = np.arange(32154, 33599 + 1)
ribosome_dic["protein_L21"] = np.arange(33600, 34876 + 1)
ribosome_dic["protein_L22"] = np.arange(34877, 35672 + 1)
ribosome_dic["protein_L23"] = np.arange(35673, 36676 + 1)
ribosome_dic["protein_L24"] = np.arange(36677, 37765 + 1)
ribosome_dic["protein_L25"] = np.arange(37766, 38734 + 1)
ribosome_dic["protein_L26"] = np.arange(38735, 39728 + 1)
ribosome_dic["protein_L27"] = np.arange(39729, 40821 + 1)
ribosome_dic["protein_L28"] = np.arange(40822, 41995 + 1)
ribosome_dic["protein_L29"] = np.arange(41996, 42458 + 1)
ribosome_dic["protein_L30"] = np.arange(42459, 43201 + 1)
ribosome_dic["protein_L31"] = np.arange(43202, 44091 + 1)
ribosome_dic["protein_L32"] = np.arange(44092, 45111 + 1)
ribosome_dic["protein_L33"] = np.arange(45112, 45962 + 1)
ribosome_dic["protein_L34"] = np.arange(45963, 46843 + 1)
ribosome_dic["protein_L35"] = np.arange(46844, 47813 + 1)
ribosome_dic["protein_L36"] = np.arange(47814, 48585 + 1)
ribosome_dic["protein_L37"] = np.arange(48586, 49267 + 1)
ribosome_dic["protein_L38"] = np.arange(49268, 49880 + 1)
ribosome_dic["protein_L39"] = np.arange(49881, 50317 + 1)
ribosome_dic["protein_L40"] = np.arange(50318, 50735 + 1)
ribosome_dic["protein_L41"] = np.arange(50736, 50969 + 1)
ribosome_dic["protein_L42"] = np.arange(50970, 51817 + 1)
ribosome_dic["protein_L43"] = np.arange(51818, 52512 + 1)
ribosome_dic["protein_P0"] = np.concatenate(
    (np.arange(52513, 53359 + 1), np.arange(53360, 53479 + 1))
)
# Starting small subunit
ribosome_dic["GNBPSBLP"] = np.arange(53480, 55924 + 1)
ribosome_dic["protein_S0"] = np.arange(55925, 57536 + 1)
ribosome_dic["protein_S1"] = np.arange(57537, 59245 + 1)
ribosome_dic["protein_S2"] = np.arange(59246, 60880 + 1)
ribosome_dic["protein_S3"] = np.arange(60881, 62614 + 1)
ribosome_dic["protein_S4"] = np.arange(62615, 64683 + 1)
ribosome_dic["protein_S5"] = np.arange(64684, 66293 + 1)
ribosome_dic["protein_S6"] = np.arange(66294, 68113 + 1)
ribosome_dic["protein_S7"] = np.arange(68114, 69594 + 1)
ribosome_dic["protein_S8"] = np.concatenate(
    (np.arange(69595, 70571 + 1), np.arange(70572, 71084 + 1))
)
ribosome_dic["protein_S9"] = np.arange(71085, 72578 + 1)
ribosome_dic["protein_S10"] = np.arange(72579, 73395 + 1)
ribosome_dic["protein_S11"] = np.arange(73396, 74640 + 1)
ribosome_dic["protein_S12"] = np.arange(74641, 75575 + 1)
ribosome_dic["protein_S13"] = np.arange(75576, 76768 + 1)
ribosome_dic["protein_S14"] = np.arange(76769, 77710 + 1)
ribosome_dic["protein_S15"] = np.arange(77711, 78701 + 1)
ribosome_dic["protein_S16"] = np.arange(78702, 79807 + 1)
# End of bundle 1 - original PDB files
ribosome_dic["protein_S17"] = np.arange(79808, 80772 + 1)
ribosome_dic["protein_S18"] = np.arange(80773, 81965 + 1)
ribosome_dic["protein_S19"] = np.arange(81966, 83078 + 1)
ribosome_dic["protein_S20"] = np.arange(83079, 83934 + 1)
ribosome_dic["protein_S21"] = np.arange(83935, 84619 + 1)
ribosome_dic["protein_S22"] = np.arange(84620, 85641 + 1)
ribosome_dic["protein_S23"] = np.arange(85642, 86763 + 1)
ribosome_dic["protein_S24"] = np.arange(86764, 87837 + 1)
ribosome_dic["protein_S25"] = np.arange(87838, 88400 + 1)
ribosome_dic["protein_S26"] = np.arange(88401, 89169 + 1)
ribosome_dic["protein_S27"] = np.arange(89170, 89780 + 1)
ribosome_dic["protein_S28"] = np.arange(89781, 90278 + 1)
ribosome_dic["protein_S29"] = np.arange(90279, 90722 + 1)
ribosome_dic["protein_S30"] = np.arange(90723, 91197 + 1)
ribosome_dic["protein_S31"] = np.arange(91198, 91695 + 1)
# End of proteins
ribosome_dic["ribosomal_RNA_18S"] = np.arange(91696, 129644 + 1)
# End of bundle 2 - original PDB files
ribosome_dic["ribosomal_RNA_25S"] = np.concatenate(
    (
        np.arange(129645, 139292 + 1),
        np.arange(139293, 171354 + 1),
        np.arange(171355, 172453 + 1),
        np.arange(172454, 200260 + 1),
    )
)
ribosome_dic["ribosomal_RNA_58S"] = np.arange(200261, 203614 + 1)
ribosome_dic["ribosomal_RNA_5S"] = np.arange(203615, 206194 + 1)
ribosome_dic["messengerRNA"] = np.arange(206195, 206389 + 1)


def finding_chain_by_serial(serial):
    """Function will return and print from which chain a given atom (using \
    it serial number) belongs in the ribosome.
    Input: serial number.
    Output: String with respective chain.
    """
    for i in list(ribosome_dic.keys()):
        if serial in ribosome_dic[i]:
            return str(i)
    return "You provided an invalid serial number"


def allcontactsfrom(contacts, origin):
    """
    Function to extract how many contacts have any atom in origin (1D atoms \
    list).
    """
    return contacts[np.isin(contacts[:, 0:2], origin).any(axis=1)]


def extractinter(contacts, first_set, second_set):
    """
    Function to extract contacts where one atom must appears in the first set \
    and the another in the second set or vice versa.
    """
    return contacts[
        np.logical_or(
            np.logical_and(
                np.isin(contacts[:, 0], first_set),
                np.isin(contacts[:, 1], second_set),
            ),
            np.logical_and(
                np.isin(contacts[:, 1], first_set),
                np.isin(contacts[:, 0], second_set),
            ),
        )
    ]


def extractintra(contacts, first_set, second_set):
    """
    Function to extract contacts where both atoms must appears either in the \
    first set or in the second set.
    """
    return contacts[
        np.logical_or(
            np.logical_and(
                np.isin(contacts[:, 0], first_set),
                np.isin(contacts[:, 1], first_set),
            ),
            np.logical_and(
                np.isin(contacts[:, 1], second_set),
                np.isin(contacts[:, 0], second_set),
            ),
        )
    ]


def eudist(a, given_contacts, coordinates):
    """
    Function to evaluate the distance between two atoms given by contacts[a] \
    using given coordinates.
    """
    # To match the serial number given and python indexes.
    i, j = np.subtract(given_contacts[:, 0:2][a].astype(int), 1)
    return np.linalg.norm(np.subtract(coordinates[i], coordinates[j]))


def parallelizing_functions(
    n_util_cores, given_function, size_variable, *other_constant_variables
):
    """
    Function to parallelize a given function using a chosen number of cores.
    The size_variable should be iterable. Preferred numpy array.
    """
    pool = multiprocessing.Pool(n_util_cores)
    results = []
    results = pool.starmap(
        given_function,
        [
            (iter_variable, size_variable, *other_constant_variables)
            for iter_variable, b in enumerate(size_variable)
        ],
    )
    pool.close()
    #    if np.less(np.size(np.shape(results)), 2):
    #        return results
    #    else:
    #        return np.concatenate((results), axis=0)
    return np.concatenate((results), axis=0)


def distances_comp(given_pair_list, coordinates, f=0.096):
    """
    Function to evaluate the distances of all pairs given, using the \
    coordinates. All returned distances will correspond to 0.96 times the \
    original distance found using the coordinates. The result will be given \
    in nanometers if the coordinates are in angstroms.
    It uses the outside parallelize function.
    """
    cores = multiprocessing.cpu_count()
    distances = parallelizing_functions(
        cores, eudist, given_pair_list, coordinates
    )
    rs = np.empty(shape=(0, 1))
    rs = np.multiply(
        np.reshape(
            np.asarray(distances), (np.shape(np.asarray(distances))[0], 1)
        ),
        f,
    )
    return rs


def distances_comp1(contacts, coordinates, f=0.096):
    """
    Function to evaluate the distances of all pairs given, using the \
    coordinates. All returned distances will correspond to 0.96 times the \
    original distance found using the coordinates. The result will be given \
    in nanometers if the coordinates are in angstroms.
    """
    cores = multiprocessing.cpu_count()
    # all returned distances will be scaled by factor f below and given in nm
    # f = 0.096
    pool = multiprocessing.Pool(cores)
    distances = []
    distances = pool.starmap(
        eudist, [(a, contacts, coordinates) for a, b in enumerate(contacts)]
    )
    pool.close()
    rs = np.empty(shape=(0, 1))
    rs = np.multiply(
        np.reshape(
            np.asarray(distances), (np.shape(np.asarray(distances))[0], 1)
        ),
        f,
    )
    # rs = rs.astype(precision)
    return rs


def sortcontacts(newcontacts):
    """
    Function to sort the contacts by their indices. The first column will \
    have priority.
    """
    # First sort doesn't need to be stable.
    newcontacts = newcontacts[newcontacts[:, 1].argsort()]
    newcontacts = newcontacts[newcontacts[:, 0].argsort(kind="mergesort")]
    return newcontacts


def fromlistto1d(inputlist):
    """
    Function to change the input (a list, an array or a sequence) in a column \
    vector.
    """
    inputlist = np.asarray(inputlist)
    return inputlist.reshape(-1, 1)


def rprime(rM, rS):
    """
    Function to evaluate r' given the distances in both structures.
    r' is defined as the new Lennard-Jones potential minimum where the energy\
    "penalty" would be the same for each old minima given.
    """
    return np.power(
        np.divide(
            np.multiply(2, np.subtract(np.power(rM, -6), np.power(rS, -6))),
            np.subtract(np.power(rM, -12), np.power(rS, -12)),
        ),
        1 / 6,
    )


def X(rp, rM):
    """
    Function to evaluate the energy "penalty" between the two minima given. \
    Usually r' is the first set of minima to be compared with the former ones.
    """
    return np.add(
        1,
        np.subtract(
            np.power(np.divide(rp, rM), 12),
            np.multiply(2, np.power(np.divide(rp, rM), 6)),
        ),
    )


def no_scaled_contacts(rp, rM, contacts):
    """
    Function to evaluate new A and B coefficients for rp (rprime) using \
    just the original distance (rM) and the contacts coefficients.
    """
    return np.concatenate(
        (
            contacts[:, 0:3],
            np.multiply(
                np.multiply(np.power(np.divide(rp, rM), 6), 1),
                fromlistto1d(contacts[:, 3]),
            ),
            np.multiply(
                np.multiply(np.power(np.divide(rp, rM), 12), 1),
                fromlistto1d(contacts[:, 4]),
            ),
        ),
        axis=1,
    )


def scale_factor(D, x):
    """
    Function to return the ratio between the first and second variables. \
    It was created to calculate the scale factor (SF) to change the contacts \
    what vary above the threshold - in this case, they will have a $Delta U'$ \
    equal $Delta U cdot epsilon_contacts$.
    """
    return np.divide(D, x)


def scaled_contacts(SF, rp, rM, contacts):
    """
    Function to evaluate new A and B coefficients for rp (rprime) using \
    the original distance (rM), the contacts coefficients and scaling the \
    epsilon by the scaling factor. The output contacts would have a new \
    minima (rprime) and the difference between the energy at rp and rM should \
    be D times epsilon, if SF = scale_factor(D, X)
    """
    return np.concatenate(
        (
            contacts[:, 0:3],
            np.multiply(
                np.multiply(np.power(np.divide(rp, rM), 6), SF),
                fromlistto1d(contacts[:, 3]),
            ),
            np.multiply(
                np.multiply(np.power(np.divide(rp, rM), 12), SF),
                fromlistto1d(contacts[:, 4]),
            ),
        ),
        axis=1,
    )


def contactsappend(*parts):
    """
    Function to append arrays of any shape with option axis=0. It will break \
    if the given arrays do not have the same dimensionality.
    """
    for q in parts:
        if np.not_equal(np.shape(q)[1], np.shape(parts[0])[1]):
            break
    p = np.empty(shape=(0, np.shape(parts[0])[1]))
    for q in parts:
        p = np.append(p, q, axis=0)
    return p


def splitcontacts(t_threshold, X_compare):
    """
    Function to split contacts based in a threshold. The first returned \
    boolean vector will show values where the value given is less or equal \
    the threshold and the second vector will return the boolean vector with \
    true for the values above the threshold.
    """
    # indexes of contacts that match the "t" criteria
    idxb = np.less_equal(np.abs(X_compare), np.abs(t_threshold))[:, 0]
    # indexes of contacts that does not match the "t" criteria
    idxa = np.greater(np.abs(X_compare), np.abs(t_threshold))[:, 0]
    return idxb, idxa


def changecontacts(BT, t, scale, contactsM, coordinatesM, coordinatesS):
    """
    Function to change a given list of contacts. All the contacts will have a\
    new minima in rprime. Also, the contacts below the threshold will be \
    scaled by the scaling factor given if required by uses of a flag in the \
    first variable. The output will be two sets of contacts, the changed and \
    scaled in the first and the no-scaled changed in the second.
    """
    rM = distances_comp1(contactsM, coordinatesM)
    rS = distances_comp1(contactsM, coordinatesS)
    rp = rprime(rM, rS)
    XM = X(rp, rM)
    idxb, idxa = splitcontacts(t, XM)
    contactsM_rp_no_scaled = no_scaled_contacts(rp, rM, contactsM)
    if np.equal(BT, True):
        contactsM_rp_scaled = scaled_contacts(scale, rp, rM, contactsM)
        final = contactsappend(
            contactsM_rp_no_scaled[idxb], contactsM_rp_scaled[idxa]
        )
        no_used = []
    else:
        final = contactsM_rp_no_scaled[idxb]
        no_used = contactsM[idxa]
    final = sortcontacts(final)
    return final, no_used


def fixing_output_file(filename, header, folder_name=""):
    """
    Function to fixing the output file adding the given header.
    """
    with open(folder_name + filename, "r") as contents:
        save = contents.read()
    with open(folder_name + filename, "w") as contents:
        contents.write(header)
    with open(folder_name + filename, "a") as contents:
        contents.write(save)
        contents.write("\n")
    return


def merging_contacts_exclusions(name, *filename, folder_name=""):
    """
    Function to merge the contacts and the exclusions definition in a final \
    file.
    """
    with open(str(folder_name + name), "w") as outfile:
        for i in filename:
            with open(folder_name + i, "r") as contents:
                for line in contents:
                    outfile.write(line)
    return


def exporting_contacts_exclusions(name, contactsMfinal, folder_name=""):
    """
    Function to save the contacts and exclusions of a force fiedl in the same \
    file.
    """
    saving_contacts("processed_" + str(name), contactsMfinal, folder_name)
    saving_exclusions("exclusions_" + str(name), contactsMfinal, folder_name)
    merging_contacts_exclusions(
        str(name),
        "processed_" + str(name),
        "exclusions_" + str(name),
        folder_name=folder_name,
    )
    pass


def saving_contacts(filename, contactsMfinal, folder_name=""):
    """
    Function to save the formatted contacts with its header.
    """
    np.savetxt(
        folder_name + filename,
        contactsMfinal,
        fmt=["%d"] * 3 + ["%10.9e"] * 2,
        delimiter="\t",
    )
    fixing_output_file(
        filename, "[ pairs ] \n;ai\taj\ttype\tA \t \tB \n", folder_name
    )
    pass


def saving_bond_like_contacts(filename, contactsMfinal, folder_name=""):
    """
    Function to save the contacts changed to work as bonds type6. A commented\
    header is included.
    """
    np.savetxt(
        folder_name + filename,
        contactsMfinal,
        fmt=["%d"] * 3 + ["%10.9e"] * 2,
        delimiter="\t",
    )
    fixing_output_file(
        filename,
        ";[ bond_like_contacts ] \n;ai\taj\tfunc\tr0(nm) \t \tKb \n",
        folder_name,
    )
    pass


def bond_like_contacts(given_contacts, spring_constant):
    """
    Function to get a list of given contacts and write them as harmonic bonds\
    with a given spring-like constant, keeping the former minimum for each \
    pair.
    """
    # to get the distances from the contacts coefficients
    r = np.power(
        np.multiply(np.divide(given_contacts[:, 4], given_contacts[:, 3]), 2),
        1 / 6,
    )
    funct = np.multiply(np.ones(shape=(np.shape(r))), 6)
    k_b = np.multiply(np.ones(shape=(np.shape(r))), spring_constant)
    return np.concatenate(
        (
            given_contacts[:, 0:2],
            fromlistto1d(funct),
            fromlistto1d(r),
            fromlistto1d(k_b),
        ),
        axis=1,
    )


def saving_exclusions(filename, contactsMfinal, folder_name=""):
    """
    Function to generate the exclusions list from the contacts list.
    """
    exclusions = np.asarray(contactsMfinal[:, 0:2], dtype=int)
    np.savetxt(
        folder_name + filename, exclusions, fmt=["%d"] * 2, delimiter="\t"
    )
    fixing_output_file(filename, "[ exclusions ] \n;ai\taj \n", folder_name)
    return


def load_contacts(filename):
    """
    Function to load the contacts from a file beginning with "[ pairs ]".
    """
    return np.genfromtxt(filename, skip_header=2)


def Ap(rp, epsilon_contacts, precision=np.float):
    """
    Function returns the A LJ coefficient for a given rprime
    """
    return np.multiply(
        np.multiply(np.power(rp, 6, dtype=precision), epsilon_contacts), 2
    )


def Bp(rp, epsilon_contacts, precision=np.float):
    """
    Function returns the B LJ coefficient for a given rprime
    """
    return np.multiply(
        np.multiply(np.power(rp, 12, dtype=precision), epsilon_contacts), 1
    )


def newContacts(contactsM, epsilon_contacts, precision=np.float):
    """
    Function to evaluate the new contacts coefficients A and B and assigning\
    the new epsilon_contacts given. This function does not change the \
    contacts distances.
    """
    A = np.multiply(
        np.multiply(
            np.divide(contactsM[:, 4], contactsM[:, 3], dtype=precision),
            epsilon_contacts,
        ),
        4,
    )
    B = np.multiply(
        np.power(
            np.multiply(
                np.divide(contactsM[:, 4], contactsM[:, 3], dtype=precision),
                2,
            ),
            2,
        ),
        epsilon_contacts,
    )
    one = np.ones(shape=(np.shape(B)))
    return np.concatenate(
        (
            contactsM[:, 0:2],
            fromlistto1d(one),
            fromlistto1d(A),
            fromlistto1d(B),
        ),
        axis=1,
    )


def scaling_contacts(SF, contactsM):
    """
    Function to multiply the contacts coefficients by the given scale factor.
    """
    newA = np.multiply(contactsM[:, 3], SF)
    newB = np.multiply(contactsM[:, 4], SF)
    one = np.ones(shape=(np.shape(newB)))
    return np.concatenate(
        (
            contactsM[:, 0:2],
            fromlistto1d(one),
            fromlistto1d(newA),
            fromlistto1d(newB),
        ),
        axis=1,
    )


def evaluatenewAB(
    BT,
    t,
    scale,
    n,
    contacts_indexes,
    coordinatesM,
    coordinatesS,
    epsilon_contacts,
):
    """
    Function to evaluate all contacts coefficients A and B whose pairs were \
    given using rprime and the given epsilon_contacts. The values below the \
    threshold would be scaled using scale factor, if asked. If a list of \
    contacts with the A and B coefficients is given, the no used contacts \
    will be returned with their former coefficients. Otherwise, it will be \
    given the coefficients evaluated with rprime.
    """
    rM = distances_comp1(contacts_indexes, coordinatesM)
    rS = distances_comp1(contacts_indexes, coordinatesS)
    rp = rprime(rM, rS)
    XM = X(rp, rM)
    idxb, idxa = splitcontacts(t, XM)
    A = Ap(rp, epsilon_contacts, precision=np.longdouble)
    B = Bp(rp, epsilon_contacts, precision=np.longdouble)
    one = np.ones(shape=(np.shape(B)))
    evaluated = np.concatenate(
        (
            contacts_indexes[:, 0:2],
            fromlistto1d(one),
            fromlistto1d(A),
            fromlistto1d(B),
        ),
        axis=1,
    )
    if np.equal(BT, True):
        scaled = scaling_contacts(scale, evaluated)
        final = contactsappend(evaluated[idxb], scaled[idxa])
        no_used = []
    else:
        final = evaluated[idxb]
        if np.greater(np.shape(contacts_indexes)[1], 2):
            no_used = contacts_indexes[idxa]
        else:
            no_used = evaluated[idxa]
        no_used = sortcontacts(no_used)
    final = sortcontacts(final)
    return final, no_used


def evaluate_AB_given_coordinates(
    contacts_indexes, coordinatesM, epsilon_contacts
):
    """
    Function to evaluate all given contacts. Regardless the former \
    coefficients, the output will use both the coordinates and epsilon.\
    given.
    """
    rM = distances_comp1(contacts_indexes, coordinatesM)
    A = Ap(rM, epsilon_contacts, precision=np.longdouble)
    B = Bp(rM, epsilon_contacts, precision=np.longdouble)
    one = np.ones(shape=(np.shape(B)))
    return np.concatenate(
        (
            contacts_indexes[:, 0:2],
            fromlistto1d(one),
            fromlistto1d(A),
            fromlistto1d(B),
        ),
        axis=1,
    )


def sortdihedrals(newdihedrals):
    """
    Function to sort the dihedrals by their indices, first, second, third \
    and fourth columns respectively.
    """
    newdihedrals = newdihedrals[newdihedrals[:, 7].argsort()]
    newdihedrals = newdihedrals[newdihedrals[:, 4].argsort(kind="mergesort")]
    newdihedrals = newdihedrals[newdihedrals[:, 3].argsort(kind="mergesort")]
    newdihedrals = newdihedrals[newdihedrals[:, 2].argsort(kind="mergesort")]
    newdihedrals = newdihedrals[newdihedrals[:, 1].argsort(kind="mergesort")]
    newdihedrals = newdihedrals[newdihedrals[:, 0].argsort(kind="mergesort")]
    return newdihedrals


def manual_parsing(filename):
    """
    Function to parse a given file with the dihedrals definition; replacing \
    spaces and double tabs for simple tabs. Also, adding zero in the improper\
    dihedrals.
    """
    out = []
    lengths = []
    with open(filename, "r") as fin:
        for line in fin:
            l = line.replace(" ", "\t")
            l = l.translate({ord("\n"): None})
            l = l.translate({ord("\r"): None})
            while "\t\t" in l:
                l = l.replace("\t\t", "\t")
            l = l.split("\t")
            out.append(l)
            lengths.append(len(l))
    lim = np.max(lengths)
    for l in out:
        while len(l) < lim:
            l.append("0")
    return np.array(out)


def deleting_zeros(filename, folder_name=""):
    """
    Function to delete the zeros assigned to improper dihedrals before save \
    them in a file.
    """
    with open(folder_name + filename, "r") as contents:
        save = contents.read()
        save = save.replace("\t0\n", "\n")
        save = save.replace("\t0\t", "\t")
    with open(folder_name + filename, "w") as contents:
        contents.write(save)
    pass


def saving_dihedrals(filename, dihedrals, folder_name=""):
    """
    Function to save the dihedrals with the correct header.
    """
    np.savetxt(
        folder_name + filename,
        dihedrals,
        fmt=["%d"] * 5 + ["%10.9e"] + ["%10.9e"] + ["%d"] * 1,
        delimiter="\t",
    )
    deleting_zeros(filename, folder_name=folder_name)
    fixing_output_file(
        filename,
        "[ dihedrals ]\n;ai\taj\tak\tal\tfunc\tphi0(deg)\tKd\t\tmult \n",
        folder_name=folder_name,
    )
    pass


def change_dihedrals(dihedrals_to_change, dihedrals_base, modified_atoms):
    """
    Function to replace the dihedrals involving the atoms given by \
    modified_atoms list and assign their values to the first set based on the\
    second.
    """
    dihedrals_to_change = sortdihedrals(dihedrals_to_change)
    dihedrals_base = sortdihedrals(dihedrals_base)
    if (np.equal(dihedrals_to_change[:, 0:4], dihedrals_base[:, 0:4])).all():
        index_to_change = np.any(
            np.isin(dihedrals_to_change[:, 0:4], modified_atoms), axis=1
        )
        #         index_to_keep = np.all(np.isin(dihedrals_to_change[:, 0:4], \
        #                                        modified_atoms, invert=True), axis=1)
        dihedrals_to_change[index_to_change] = dihedrals_base[index_to_change]
    return dihedrals_to_change


def splitting_dihedrals(dihedrals, precision=np.float):
    """
    Function to separate the dihedrals. Returning four sets of dihedrals: \
    the impropers, the rings (dihedrals funct 4 with multiplicity 2), \
    proper with multiplicity 1 and 3, in this order.
    """
    dihedrals = dihedrals.astype(precision)
    return (
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 2), np.equal(dihedrals[:, 7], 0)
            )
        ],
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 4), np.equal(dihedrals[:, 7], 2)
            )
        ],
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 1), np.equal(dihedrals[:, 7], 1)
            )
        ],
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 1), np.equal(dihedrals[:, 7], 3)
            )
        ],
    )


def splitting_dihedrals_old(dihedrals, precision=np.float):
    """
    Function to separate the dihedrals. Returning three sets of dihedrals, \
    the impropers, the rings (dihedrals funct 2 with multiplicity 2), \
    proper with multiplicity 1 and 3, in this order.
    """
    dihedrals = dihedrals.astype(precision)
    return (
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 2), np.equal(dihedrals[:, 7], 0)
            )
        ],
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 1), np.equal(dihedrals[:, 7], 1)
            )
        ],
        dihedrals[
            np.logical_and(
                np.equal(dihedrals[:, 4], 1), np.equal(dihedrals[:, 7], 3)
            )
        ],
    )


def averaged_dihedrals(dihedralsM, dihedralsS, precision=np.float):
    """
    Function to evaluate the average of proper dihedral angles without \
    changing the energy term
    """
    sub0M, sub2M, sub1M, sub3M = splitting_dihedrals(dihedralsM, precision)
    sub0S, sub2S, sub1S, sub3S = splitting_dihedrals(dihedralsS, precision)
    newsub1M = treat_propdihedrals(N, sub1M, sub1S)
    newsub1S = treat_propdihedrals(N, sub1S, sub1M)
    newsub3M = np.concatenate(
        (
            newsub1M[:, 0:5],
            fromlistto1d(np.multiply(newsub1M[:, 5], 3)),
            sub3M[:, 6:8],
        ),
        axis=1,
    )
    newsub3S = np.concatenate(
        (
            newsub1S[:, 0:5],
            fromlistto1d(np.multiply(newsub1S[:, 5], 3)),
            sub3S[:, 6:8],
        ),
        axis=1,
    )
    return (
        sortdihedrals(contactsappend(sub0M, sub2M, newsub1M, newsub3M)),
        sortdihedrals(contactsappend(sub0S, sub2S, newsub1S, newsub3S)),
    )


def averaged_all_dihedrals(dihedralsM, dihedralsS, precision=np.float):
    """
    Function to evaluate the average of all dihedrals, separately. Return \
    them in one array, sorted and concatenated.
    """
    # splitting the dihedrals
    sub0M, sub2M, sub1M, sub3M = splitting_dihedrals(dihedralsM, precision)
    sub0S, sub2S, sub1S, sub3S = splitting_dihedrals(dihedralsS, precision)
    # evaluating the average for the proper - funct 1 multiplicity 1
    newsub1M = treat_propdihedrals(N, sub1M, sub1S)
    newsub1S = treat_propdihedrals(N, sub1S, sub1M)
    # evaluating the average for the impropers - funct 2 multiplicity 0
    newsub0M = treat_impropdihedrals(sub0M, sub0S)
    newsub0S = treat_impropdihedrals(sub0S, sub0M)
    # evaluating the average for the impropers - funct 4 multiplicity 2
    newsub2M = treat_impropdihedrals(sub2M, sub2S)
    newsub2S = treat_impropdihedrals(sub2S, sub2M)
    # evaluating the propers with funct 1 and multiplicity 3
    newsub3M = np.concatenate(
        (
            newsub1M[:, 0:5],
            fromlistto1d(np.multiply(newsub1M[:, 5], 3)),
            sub3M[:, 6:8],
        ),
        axis=1,
    )
    newsub3S = np.concatenate(
        (
            newsub1S[:, 0:5],
            fromlistto1d(np.multiply(newsub1S[:, 5], 3)),
            sub3S[:, 6:8],
        ),
        axis=1,
    )
    return (
        sortdihedrals(contactsappend(newsub0M, newsub2M, newsub1M, newsub3M)),
        sortdihedrals(contactsappend(newsub0S, newsub2S, newsub1S, newsub3S)),
    )


def getdihedrals(filename, precision=np.float):
    """
    Function to load and parse the dihedrals definition. Delimiter must be \
    tab ("\t")
    """
    temp = manual_parsing(filename)[2:]
    final_dihedrals = []
    # np.savetxt("_dihedral_temp.txt", temp)
    # final_dihedrals = np.genfromtxt("_dihedral_temp.txt")
    # os.remove("_dihedral_temp.txt")
    for i in np.asarray(temp):
        if np.equal(np.count_nonzero(i), np.size(i)):
            final_dihedrals.append(i)
    return np.asarray(final_dihedrals, dtype=precision)[:, :8]


def evaluating_mean_angle(i, sub1M, sub1S):
    """
    Function to evaluate the mean angle for each proper dihedral, considering\
    their periodicidy.
    """
    mean = 0
    if np.less(np.abs(np.subtract(sub1S[i][5], sub1M[i][5])), 180):
        mean = (sub1S[i][5] + sub1M[i][5]) / 2.0
    else:
        mean = (sub1S[i][5] + (sub1M[i][5] - 360)) / 2.0
        if np.less(mean, 0):
            mean = mean + 360
    return mean


def normalize_angle(angle):
    """Function to get an angle (or a list of angles) and return their proper\
    positive value between 0 and 360."""
    return np.mod(angle, 360)


def evaluating_mean_angle_new(i, sub1M, sub1S):
    """
    Function to evaluate the mean angle for each proper dihedral, considering\
    their periodicidy.
    """
    mean = 0
    angle_M = normalize_angle(sub1M[i][5])
    angle_S = normalize_angle(sub1S[i][5])
    if np.less(np.abs(np.subtract(angle_M, angle_S)), 180):
        mean = (angle_M + angle_S) / 2.0
    else:
        mean = (angle_M + (angle_S - 360)) / 2.0
    return normalize_angle(mean)


def load_rmsf(filename):
    """
    Function to load the rmsf.xvg files generated with GROMACS.
    """
    return np.genfromtxt(filename, skip_header=16)


def load_rmsd(filename):
    """
    Function to load the rmsd.xvg files generated with GROMACS.
    """
    return np.genfromtxt(filename, skip_header=17)


def treat_propdihedrals(Na, sub1M, sub1S):
    """
    Function to calculate the proper dihedrals, rescaling the epsilon based on\
    total number of dihedrals.
    """
    # Get share of energy in proper dihedrals
    SUMP = np.divide(Na, 3)
    norm = np.divide(sub1M[:, 6], sub1M[:, 6][0])
    a = np.divide(SUMP, norm.sum())
    newKd = np.multiply(a, norm)
    # Evaluating the average of dihedral angles
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    mean = []
    mean = pool.starmap(
        evaluating_mean_angle_new,
        [(a, sub1M, sub1S) for a, b in enumerate(sub1M)],
    )
    return np.concatenate(
        (
            sub1M[:, 0:5],
            fromlistto1d(mean),
            fromlistto1d(newKd),
            fromlistto1d(sub1M[:, 7]),
        ),
        axis=1,
    )


def treat_impropdihedrals(subiM, subiS):
    """
    Function to evaluate the average angle of improper dihedrals.
    """
    # Evaluating the average of improper dihedral angles
    subiM = sortdihedrals(subiM)
    subiS = sortdihedrals(subiS)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    mean = []
    mean = pool.starmap(
        evaluating_mean_angle_new,
        [(a, subiM, subiS) for a, b in enumerate(subiM)],
    )
    return np.concatenate(
        (
            subiM[:, 0:5],
            fromlistto1d(mean),
            fromlistto1d(subiM[:, 6]),
            fromlistto1d(subiM[:, 7]),
        ),
        axis=1,
    )


def change_inter_unique(SF, contacts):
    """
    Function to rescale the contacts epsilon by the giving scaled factor.
    """
    A = fromlistto1d(np.multiply(contacts[:, 3], SF))
    B = fromlistto1d(np.multiply(contacts[:, 4], SF))
    return np.concatenate((contacts[:, 0:3], A, B), axis=1)


def returning_dictionary_contacts(contacts):
    """Function to return a dictionary with some contacts sets.
    Input: list of contacts
    Return: filled dictionary with these categories and the number of contacts \
    of each one:
    - "modeled_inter_L24"
    - "modeled_intra_L24"
    - "modeled_internal_L24"
    - "modeled_inter_RNA"
    - "modeled_intra_RNA"
    - "modeled_internal_RNA"
    - "corrected_inter_RNA"
    - "corrected_intra_RNA"
    - "corrected_internal_RNA
    - "corrected_inter_S1"
    - "corrected_intra_S1"
    - "corrected_internal_S1"
    - "corrected_between"
    - "intra_no_modeled"
    - "inter_no_modeled"
    - "intra_no_modified"
    - "inter_no_modified"
    """
    dic = {}
    dic["modeled_inter_L24"] = np.shape(
        extractinter(contacts, small_total, l24_modeled_f3j77)
    )[0]
    dic["modeled_intra_L24"] = np.shape(
        extractinter(contacts, large_no_modified, l24_modeled_f3j77)
    )[0]
    dic["modeled_internal_L24"] = np.shape(
        extractinter(contacts, l24_modeled_f3j77, l24_modeled_f3j77)
    )[0]
    dic["modeled_inter_RNA"] = np.shape(
        extractinter(contacts, small_total, RNA_25S_modeled_f3j78)
    )[0]
    dic["modeled_intra_RNA"] = np.shape(
        extractinter(contacts, large_no_modified, RNA_25S_modeled_f3j78)
    )[0]
    dic["modeled_internal_RNA"] = np.shape(
        extractinter(contacts, RNA_25S_modeled_f3j78, RNA_25S_modeled_f3j78)
    )[0]
    dic["corrected_inter_RNA"] = np.shape(
        extractinter(contacts, small_no_corrected, RNA_25S_corrected_f3j77)
    )[0]
    dic["corrected_intra_RNA"] = np.shape(
        extractinter(contacts, large_no_corrected, RNA_25S_corrected_f3j77)
    )[0]
    dic["corrected_internal_RNA"] = np.shape(
        extractinter(
            contacts, RNA_25S_corrected_f3j77, RNA_25S_corrected_f3j77
        )
    )[0]
    dic["corrected_inter_S1"] = np.shape(
        extractinter(contacts, large_no_corrected, S1_corrected_f3j77)
    )[0]
    dic["corrected_intra_S1"] = np.shape(
        extractinter(contacts, small_no_corrected, S1_corrected_f3j77)
    )[0]
    dic["corrected_internal_S1"] = np.shape(
        extractinter(contacts, S1_corrected_f3j77, S1_corrected_f3j77)
    )[0]
    dic["corrected_between"] = np.shape(
        extractinter(contacts, RNA_25S_corrected_f3j77, S1_corrected_f3j77)
    )[0]
    dic["intra_no_modeled"] = np.shape(
        extractintra(contacts, small_total, large_no_modeled)
    )[0]
    dic["inter_no_modeled"] = np.shape(
        extractinter(contacts, small_total, large_no_modeled)
    )[0]
    dic["intra_no_modified"] = np.shape(
        extractintra(contacts, small_no_modified, large_no_modified)
    )[0]
    dic["inter_no_modified"] = np.shape(
        extractinter(contacts, small_no_modified, large_no_modified)
    )[0]
    return dic


def comparison_core(a_list, b_list):
    """
    Core of comparison_pairs function. Just to avoid repeating lines of \
    code.
    """
    results = []
    for pair in a_list[:, [0, 1]]:
        if not np.any(
            [np.isin(pairb, pair).all() for pairb in b_list[:, [0, 1]]]
        ):
            results.append(pair)
    return np.asarray(results)


def comparing_pairs(list_a, list_b):
    """
    Function will return the pairs that appear in the one list and do \
    not appear in the another.
    Input: two list of pairs or contacts
    Output: two numpy arrays with the different pairs, if they exist. The \
    first list contains the pairs which appears just in the first given list. \
    The second list contains the pairs which appears just in the second list. \
    """
    results_ab = comparison_core(list_a, list_b)
    results_ba = comparison_core(list_b, list_a)
    return np.asarray(results_ab), np.asarray(results_ba)


def returning_multidimensional_contacts(contacts):
    """
    Function to return list of arrays with some contacts sets.
    Input: list of contacts
    Return: list of 17 arrays filled list with the following order:
    0 - "modeled_inter_L24"
    1 - "modeled_intra_L24"
    2 - "modeled_internal_L24"
    3 - "modeled_inter_RNA"
    4 - "modeled_intra_RNA"
    5 - "modeled_internal_RNA"
    6 - "corrected_inter_RNA"
    7 - "corrected_intra_RNA"
    8 - "corrected_internal_RNA
    9 - "corrected_inter_S1"
    10 - "corrected_intra_S1"
    11 - "corrected_internal_S1"
    12 - "corrected_between"
    13 - "intra_no_modeled"
    14 - "inter_no_modeled"
    15 - "intra_no_modified"
    16 - "inter_no_modified"
    """
    result = []
    result.append(extractinter(contacts, small_total, l24_modeled_f3j77))
    result.append(
        extractinter(contacts, large_no_modified, l24_modeled_f3j77)
    )
    result.append(
        extractinter(contacts, l24_modeled_f3j77, l24_modeled_f3j77)
    )
    result.append(extractinter(contacts, small_total, RNA_25S_modeled_f3j78))
    result.append(
        extractinter(contacts, large_no_modified, RNA_25S_modeled_f3j78)
    )
    result.append(
        extractinter(contacts, RNA_25S_modeled_f3j78, RNA_25S_modeled_f3j78)
    )
    result.append(
        extractinter(contacts, small_no_corrected, RNA_25S_corrected_f3j77)
    )
    result.append(
        extractinter(contacts, large_no_corrected, RNA_25S_corrected_f3j77)
    )
    result.append(
        extractinter(
            contacts, RNA_25S_corrected_f3j77, RNA_25S_corrected_f3j77
        )
    )
    result.append(
        extractinter(contacts, large_no_corrected, S1_corrected_f3j77)
    )
    result.append(
        extractinter(contacts, small_no_corrected, S1_corrected_f3j77)
    )
    result.append(
        extractinter(contacts, S1_corrected_f3j77, S1_corrected_f3j77)
    )
    result.append(
        extractinter(contacts, RNA_25S_corrected_f3j77, S1_corrected_f3j77)
    )
    result.append(extractintra(contacts, small_total, large_no_modeled))
    result.append(extractinter(contacts, small_total, large_no_modeled))
    result.append(
        extractintra(contacts, small_no_modified, large_no_modified)
    )
    result.append(
        extractinter(contacts, small_no_modified, large_no_modified)
    )
    return result


def analyzing_merged(*contacts):
    """
    Function to return all pairs that appears after concatenating the given
    contacts list.
    """
    all_selected = np.concatenate(([i for i in contacts]), axis=0)
    all_selected_unique = np.unique(all_selected[:, 0:2], axis=0)
    return all_selected_unique.astype(int)


def rmsf_analysis(filename, precision=np.longdouble):
    """
    Function to return the number of elements, average, mean square deviation \
    and the standard deviation
    """
    rmsf = load_rmsf(filename)
    rmsf = np.asarray(rmsf).astype(precision)
    mean = np.mean(rmsf, axis=0)[1]
    std = np.std(rmsf, axis=0)[1]
    square_std = np.square(std)
    return np.shape(rmsf)[0], mean, square_std, std


def search_text_criteria(filename, regex_expression, show_next):
    """
    Function will return the given line extracting values when (or after) \
    matched criteria given by regex_expression. show_next should be boolean
    """
    matched = []
    with open(filename, "r") as ifile:
        for line in ifile:
            if re.match(regex_expression, line):
                if show_next:
                    matched.append((next(ifile, "").strip()))
                else:
                    matched.append(line)
    return matched


def extract_values_log(matched_lines, precision=np.longdouble):
    """
    Function to generate the numpy array with all values after criteria. \
    Should be revised.
    """
    if not matched_lines:
        print("The search returned void.")
        print(np.shape(matched_lines))
        print(matched_lines)
        return np.asarray([])
    else:
        results_shape = (0, np.size(np.fromstring(matched_lines[0], sep=" ")))
    results = np.empty(shape=(results_shape))
    for i, j in enumerate(matched_lines):
        results = np.append(
            results,
            [np.fromstring(matched_lines[i], dtype=precision, sep=" ")],
            axis=0,
        )
    return results


# To generate a numpy array with the Potential energy given read from log file
# output: time, Coulomb-14, LJ (SR), Coulomb (SR), Potential, Kinetic En., Bond,\
# Angle, Proper Dih., Improper Dih., LJ-14
def potential_rerun(filename):
    criteria_potential = "(.*)Coulomb(.*)Potential(.*)"
    results_potential = extract_values_log(
        search_text_criteria(filename, criteria_potential, True)
    )
    criteria_improper = "(.*)Bond(.*)Angle(.*)Proper(.*)"
    results_improper = extract_values_log(
        search_text_criteria(filename, criteria_improper, True)
    )
    extract_time = "(.*)Time(.*)Lambda(.*)"
    time_results = extract_values_log(
        search_text_criteria(filename, extract_time, True)
    )
    # compare if you have the same number of time and energy values; otherwise,\
    # you discard the last energy value (it is an average over the trajectory)
    if np.equal(np.shape(results_potential)[0], np.shape(time_results)[0]):
        time_potential = np.concatenate(
            (
                fromlistto1d(time_results[:, 1]),
                results_potential,
                results_improper,
            ),
            axis=1,
        )
    elif np.less(np.shape(results_potential)[0], np.shape(time_results)[0]):
        time_potential = np.concatenate(
            (
                fromlistto1d(time_results[:, 1])[:-1],
                results_potential,
                results_improper,
            ),
            axis=1,
        )
    else:
        time_potential = np.concatenate(
            (
                fromlistto1d(time_results[:, 1]),
                results_potential[:-1],
                results_improper[:-1],
            ),
            axis=1,
        )
    return time_potential


# Extract the potential energy from a list of GROMACS log files
def total_potential_rerun(file_list):
    result = np.empty(shape=(0, 11))
    for filename in file_list:
        result = np.append(result, potential_rerun(filename), axis=0)
    # To make sure the result is in ascending order of time
    result = result[result[:, 0].argsort()]
    return result


# Spliting the energy files using given angle criteria (must have angle file and\
# energy must be corrected (x values should be divided by 10))
def splitting_energy(energyfile, anglefile, theta_min, theta_max):
    indexes_below = np.isin(
        energyfile[:, 0],
        anglefile[np.less_equal(anglefile[:, 1], theta_min)][:, 0],
    )
    indexes_above = np.isin(
        energyfile[:, 0],
        anglefile[np.greater_equal(anglefile[:, 1], theta_max)][:, 0],
    )
    below_criteria = energyfile[indexes_below]
    above_criteria = energyfile[indexes_above]
    return below_criteria, above_criteria


def read_large_txt(file, eq_steps=0, delimiter=None, dtype=None):
    """
    Function to read, clean a data file and skipping the first eq_steps.
    """
    tempfile = clean_file(file)
    with open(tempfile, "r+") as FileObj:
        nrows = sum(1 for line in FileObj)
        FileObj.seek(0)
        ncols = len(next(FileObj).split(delimiter))
        out = np.empty((nrows, ncols), dtype=dtype)
        FileObj.seek(0)
        for i, line in enumerate(FileObj):
            out[i] = line.split(delimiter)
    os.remove(tempfile)
    return out[np.int(eq_steps) :]


def clean_file(file):
    """
    Function to clean up a data file, purging any comment line.
    """
    with open(file, "r") as f:
        data = f.read()
    with open("temporary", "w") as temp:
        temp.write(data)
    for pattern in ["^@.*|^%.*|^#.*|^;.*|^!.*|^[A-z].*|^$.*"]:
        matched = re.compile(pattern).search
        with open("temporary", "r") as temp:
            with open("temporary2", "w") as outfile:
                for line in temp:
                    if not matched(line):
                        outfile.write(line)
        os.replace(outfile.name, temp.name)
        return "temporary"


def contact_checker(distances, initial_distance, threshold):
    """
    Function to make a pairwise check if each given distance is below the \
    initial distance times a threshold (also given).
    """
    return np.less_equal(distances, np.multiply(initial_distance, threshold))


# OUTDATED
# function to evaluate which contacts were formed for the i-th xtc_trajectory \
# file
# output: boolean array with shape = (nsteps, npairs)
# made to be parallelized with caution - see ram memory limit.
def check_ribosome_contacts(
    i, xtc_trajectory, pdb_file, list_pairs, initial_distance, threshold=1.5
):
    trajectory = md.load_xtc(xtc_trajectory[i], top=pdb_file)
    distance = md.compute_distances(trajectory, list_pairs)
    return contact_checker(distance, initial_distance, threshold)


def evaluate_ribosome_contacts_distances(
    i, xtc_trajectory, pdb_file, list_pairs, precision=np.longdouble
):
    """
    Function to evaluate the distances between pairs for the i-th \
    xtc_trajectory file. There is a flag to discard overlaping frames in \
    the trajectory. Output: distances for each pair.
    """
    trajectory = md.load(
        xtc_trajectory[i], discard_overlapping_frames=True, top=pdb_file
    )
    return np.asarray(
        md.compute_distances(trajectory, list_pairs), dtype=precision
    )


def evaluating_contacts_chunk(
    pdb_file, xtc_file, pairs_indexes, r_initial, threshold=1.5, chunk=10000
):
    """
    Function to evaluate the number of contacts for each given timestep.
    Input:
     pdb_file - File with your structure (PDB or GRO files for instance).
     xtc_file - Trajectory.
     pairs_indexes - Numpy array Nx2 with the pairs to be used to evaluate \
     the contacts. (The first two columns of the pairs section in the TPR file \
     without the header).
     r_initial - Initial distance for each given pair to be used as a reference.
     threshold - Value to be used as a threshold to evaluate the contacts.
     chunk - Size of each chunk in which the trajectory will be analyzed.
    Output: Nx1 numpy array with the total number of contacts for each \
     timestep.
    """
    contacts = []
    for chunk_trajectory in md.iterload(xtc_file, top=pdb_file, chunk=chunk):
        trajectory = md.compute_distances(chunk_trajectory, pairs_indexes)
        print((chunk_trajectory))
        contacts.append(
            np.sum(
                np.less_equal(trajectory, np.multiply(r_initial, threshold)),
                axis=1,
            )
        )

    contacts = np.concatenate((contacts))
    return contacts


def adding_timestep(dataset, starting=1, multiplier=1):
    """
    Function to add a timestep column (ascending integer sequence).
    """
    return np.concatenate(
        (
            np.multiply(
                multiplier,
                fromlistto1d(
                    [
                        i
                        for i in range(
                            starting, starting + np.shape(dataset)[0]
                        )
                    ]
                ),
            ),
            np.reshape(
                dataset,
                (
                    np.shape(dataset)[0],
                    np.int(np.divide(np.size(dataset), np.shape(dataset)[0])),
                ),
            ),
        ),
        axis=1,
    )


def get_coordinates_pdb(pdb, precision=np.float):
    """
    Function to extract the coordinates from a PDB file. Just take the ATOM \
    coordinates.
    Input: PDB file or filename with localization.
    Output: Numpy array with the coordinates shape=(N, 3)
    """
    coordinates = []
    with open(pdb, "r") as f:
        for line in f:
            if line.find("ATOM") >= 0:
                newline = "".join(list(line)[31:54])
                coordinates.append(np.asarray(newline.split()))
    coordinates = np.asarray(coordinates, dtype=precision)
    return coordinates


def replace_coordinates_pdb(pdb, new_coordinate, new_pdb):
    """
    Function to replace the coordinates from a PDB file. Just take the \
    ATOM coordinates.
    Input: pdb: old PDB file; new_coordinate: new coordinates to be replaced; \
    Output: the new pdb filename.
    """
    i = 0
    with open(pdb, "r+") as f:
        with open(new_pdb, "w") as g:
            for line in f:
                if line.find("ATOM") >= 0:
                    newline = line[:31] + new_coordinate[i] + line[54:]
                    g.write(newline)
                    i += 1
                else:
                    newline = line
                    g.write(newline)
    pass


def alter_coordinates_pdb(pdb_alter, pdb_base, threshold=1):
    """
    Function to change the coordinates based on a threshold. It will compare \
    the pdb_alter and pdb_base atom coordinates, get the difference between \
    them, assign zero to the differences below the given threshold and return \
    the new coordinates, with the differences added to pdb_base.
    """
    coordinates_alter = get_coordinates_pdb(pdb_alter)
    coordinates_base = get_coordinates_pdb(pdb_base)
    if np.shape(coordinates_alter == coordinates_base):
        difference = np.subtract(coordinates_alter, coordinates_base)
        difference[np.less_equal(np.abs(difference), threshold)] = 0
        new_coordinates_base = difference + coordinates_base
        new_coordinates = []
        for i, k in enumerate(new_coordinates_base):
            coords = list(
                ["{:3.3f}".format(j) for j in new_coordinates_base[i]]
            )
            f_coords = (
                "{:>7}".format(coords[0])
                + " "
                + "{:>7}".format(coords[1])
                + " "
                + "{:>7}".format(coords[2])
            )
            new_coordinates.append(f_coords)
        return new_coordinates
    else:
        print("Your provided PBDs have different number of atoms.")
        return None


def strip_pdb(pdb_alter, pdb_base, new_pdb, threshold=1):
    """
    Function to strip the PDB file.
    Set to zero absolute differences below the threshold.
    Alter the atom positions with the left differences.
    Return a new pdb, saving it by just changing the atom coordinates.
    This is the main function to alter PDB files.
    """
    new_formated = alter_coordinates_pdb(pdb_alter, pdb_base, threshold)
    replace_coordinates_pdb(pdb_base, new_formated, new_pdb)
    pass


def evaluate_r_initial(contacts, model="AA", precision=np.float):
    """
    Function to evaluate initial pairwise distances accordingly the model \
    simulated.
    Input:
     contacts_list - Array with the definitions from pairs section of the \
     forcefield (TPR file.)
     model - AA = All-Atom; CA = Carbon_alpha Coarse-Grained
    Output:
     r_initial - vector with the initial distance of each pair.
    """
    contacts = np.asarray(contacts, dtype=precision)
    if model == "CA":
        r_initial = np.power(
            np.divide(np.multiply(contacts[:, 4], 1.2), contacts[:, 3]),
            np.divide(1, 2),
        )
    elif model == "AA":
        r_initial = np.power(
            np.divide(np.multiply(contacts[:, 4], 2), contacts[:, 3]),
            np.divide(1, 6),
        )
    else:
        print("You have not provided an unimplemented model.")
        exit()
    return r_initial


def evaluating_contacts_chunk(
    pdb_file, xtc_file, pairs_indexes, r_initial, threshold=1.5, chunk=10000
):
    """
    Function to evaluate the number of contacts for each given timestep.
    Input:
     pdb_file - File with your structure (PDB or GRO files for instance).
     xtc_file - Trajectory.
     pairs_indexes - Numpy array Nx2 with the pairs to be used to evaluate \
     the contacts. (The first two columns of the pairs section in the TPR file \
     without the header).
     r_initial - Initial distance for each given pair to be used as a reference.
     threshold - Value to be used as a threshold to evaluate the contacts.
     chunk - Size of each chunk in which the trajectory will be analyzed.
    Output: Nx1 numpy array with the total number of contacts for each \
     timestep.
    """
    contacts = []
    for chunk_trajectory in md.iterload(xtc_file, top=pdb_file, chunk=chunk):
        trajectory = md.compute_distances(chunk_trajectory, pairs_indexes)
        print((chunk_trajectory))
        contacts.append(
            np.sum(
                np.less_equal(trajectory, np.multiply(r_initial, threshold)),
                axis=1,
            )
        )

    contacts = np.concatenate((contacts))
    return contacts


def return_ordered(num_1, num_2):
    """Function to get two numbers and return the pair in ascending order"""
    if num_1 == num_2:
        print("Both are the same. \n Please restart with the rigth values.")
        pair = num_1, num_2
    elif num_1 < num_2:
        pair = num_1, num_2
    else:
        pair = num_2, num_1
    return pair


# def test_overlap_pairs(pair_a, pair_b):
#     """
#     Function to test if there is an overlap between two given pairs
#     Input: Two pairs.
#     Output: Boolean - True if there is NO overlap.
#     """
#     if np.logical_and(np.equal(len(pair_a), 0), np.equal(len(pair_b), 0)):
#         print("One (or both) vector(s) is(are) empty.")
#         return None
#     return np.logical_or(np.less(pair_a[1], pair_b[0]), np.greater(pair_a[0], \
#                                                                    pair_b[1]))


def checking_intervals_overlaping(input_array):
    """
    Function to check if it has an overlap in any pair of a given boundary \
    array.
    Input: Nx2 array with boundaries
    Output: Boolean - True if there is NO overlap in any interval.
    """
    results = []
    for i in itertools.combinations(range(np.shape(input_array)[0]), 2):
        results.append(
            test_overlap_pairs(input_array[i[0]], input_array[i[1]])
        )
    return np.all(results)


def gen_contact_probability(
    pdb_file, xtc_file, pairs_indexes, r_initial, threshold=1.5, chunk=10000
):
    """
    Function to evaluate the contact probability per residue.
    Input:
     pdb_file - File with your structure (PDB or GRO files for instance).
     xtc_file - Trajectory.
     pairs_indexes - Numpy array Nx2 with the pairs to be used to evaluate \
     the contacts. (The first two columns of the pairs section in the TPR file \
     without the header).
     r_initial - Initial distance for each given pair to be used as a reference.
     threshold - Value to be used as a threshold to evaluate the contacts.
     chunk - Size of each chunk in which the trajectory will be analyzed.
    Output:
     p_q_i - QxN numpy array with the contact probability at each given total \
     contacts value for each atom/residue.
     contacs_list - numpy 1-D (Q) array with the number of contacts.
     atoms_list - numpy 1-D (N) array with the atoms/residues involved.
    """

    cutoff = np.multiply(r_initial, threshold)

    # Correcting the numbering of atoms/residues involved.
    atoms_indexes = np.unique(pairs_indexes)
    atoms_involved = np.add(atoms_indexes, 1)

    # Correcting the contacts value and its correspondent index.
    contacts_indexes = np.arange(np.shape(pairs_indexes)[0] + 1)

    # Initializing the contacts involved. Is expected the total number of \
    # contacts is formed at least in the first frame.
    contacts_involved = np.asarray([np.shape(pairs_indexes)[0]])

    # Initializing the results array
    results = np.zeros(
        (np.shape(contacts_indexes)[0], np.shape(r_initial)[0])
    )

    # The last number of frames will store the total number for sanity check.
    n_frames = np.zeros(np.shape(contacts_indexes)[0] + 1)
    for chunk_trajectory in md.iterload(xtc_file, top=pdb_file, chunk=chunk):
        trajectory = md.compute_distances(chunk_trajectory, pairs_indexes)
        print(chunk_trajectory)
        # Getting the number of frames of each chunk and adding to the total
        n_frames[-1] += np.shape(trajectory)[0]
        below_threshold = np.less_equal(trajectory, cutoff)
        # Generate a matrix with 1 where contacts are formed.
        num_below_threshold = np.multiply(below_threshold, 1)
        # (number of contact per timestep)
        contacts_time = np.sum(num_below_threshold, axis=1)
        # Evaluating the contacts formed.
        contacts_involved = np.unique(
            np.concatenate((contacts_time, contacts_involved))
        )
        # Iterating over the number of contacts found.
        # n_frames receive number of frames found with Q contacts
        for i in contacts_indexes:
            idx = np.equal(contacts_time, i)
            n_frames[i] += idx.sum()
            results[i] += np.sum(num_below_threshold[idx], axis=0)

    # To normalize all the probabilities in each dimension after all pieces are\
    # read.
    for i in contacts_involved:
        results[i] = np.nan_to_num(np.divide(results[i], n_frames[i]))

    # Extracting nonzero results
    results_nz = results[contacts_involved]

    # Checking if all individual probabilities are normalized.
    assert np.less_equal(np.max(results_nz), 1)

    # Initiaizing the P(Q,i)
    p_q_i = np.zeros(
        (np.shape(contacts_involved)[0], np.shape(atoms_involved)[0])
    )
    # The probability for each atom is given multiplying the probability of all\
    # pairs with this atom.
    for i, atom in enumerate(atoms_indexes):
        idx_atom = np.isin(pairs_indexes, atom).any(axis=1)
        p_q_i[:, i] += np.sum(results_nz[:, idx_atom], axis=1)
        # normalization over the number of contacts with each given atom/residue
        p_q_i[:, i] = np.divide(p_q_i[:, i], idx_atom.sum())
    # Sanity check of number of frames read
    assert np.less_equal(np.sum(n_frames[:-1]), n_frames[-1])

    return p_q_i, contacts_involved, atoms_involved


def phi_i(
    pdb_file,
    xtc_file,
    pairs_indexes,
    r_initial,
    boundaries,
    threshold=1.5,
    chunk=10000,
):
    """
    Function to evaluate the phi-value for each atom/residue.
    Input:
     pdb_file - File with your structure (PDB or GRO files, for instance).
     xtc_file - Trajectory.
     pairs_indexes - Numpy array Nx2 with the pairs to be used to evaluate \
     the contacts. (The first two columns of the pairs section in the TPR file \
     without the header).
     r_initial - Initial distance for each given pair to be used as a reference.
     threshold - Value to be used as a threshold to evaluate the contacts.
     chunk - Size of each chunk in which the trajectory will be analyzed.
    Output
     phi - Numpy array with phi_values as a function of i - (atom/residue)
    """

    cutoff = np.multiply(threshold, r_initial)

    results = np.zeros((np.shape(boundaries)[0], np.shape(r_initial)[0]))

    # Correcting the numbering of atoms/residues involved.
    atoms_indexes = np.unique(pairs_indexes)
    atoms_involved = np.add(atoms_indexes, 1)

    # The last number of frames will store the total number for sanity check.
    n_frames = np.zeros(np.shape(boundaries)[0] + 1)
    for chunk_trajectory in md.iterload(xtc_file, top=pdb_file, chunk=chunk):
        trajectory = md.compute_distances(chunk_trajectory, pairs_indexes)
        print(chunk_trajectory)
        # Getting the number of frames of each chunk and adding to the total
        n_frames[-1] += np.shape(trajectory)[0]
        below_threshold = np.less_equal(trajectory, cutoff)
        # Generate a matrix with 1 where contacts are formed.
        num_below_threshold = np.multiply(below_threshold, 1)
        # (number of contact per timestep)
        contacts_time = np.sum(num_below_threshold, axis=1)
        for i, pair in enumerate(boundaries):
            idx = np.logical_and(
                np.greater_equal(contacts_time, pair[0]),
                np.less_equal(contacts_time, pair[1]),
            )
            n_frames[i] += idx.sum()
            results[i] += np.sum(num_below_threshold[idx], axis=0)

    # To normalize all the probabilities in each dimension after all pieces are\
    # read
    for i, j in enumerate(boundaries):
        results[i] = np.divide(results[i], n_frames[i])

    # Checking if all individual probabilities are normalized.
    assert np.less_equal(np.max(results), 1)

    # Initiaizing the phi-values
    pij_transition_unfolded = np.zeros(np.shape(atoms_involved))
    pij_folded_unfolded = np.zeros(np.shape(atoms_involved))
    # pij_folded = np.zeros(np.shape(atoms_involved))

    # Evaluating both parts of the phi fraction for each atom/residue.
    for i, atom in enumerate(atoms_indexes):
        idx_atom = np.isin(pairs_indexes, atom).any(axis=1)
        pij_transition_unfolded[i] += np.sum(
            np.subtract(results[1, idx_atom], results[0, idx_atom])
        )
        pij_folded_unfolded[i] += np.sum(
            np.subtract(results[2, idx_atom], results[0, idx_atom])
        )

    # Sanity check of number of frames read
    assert np.less_equal(np.sum(n_frames[:-1]), n_frames[-1])

    phi = np.nan_to_num(
        np.divide(pij_transition_unfolded, pij_folded_unfolded)
    )

    # # This return will give phi(i) where i is the atom/residue number given \
    # # by the forcefield.
    return np.concatenate(
        (fromlistto1d(atoms_involved), fromlistto1d(phi)), axis=1
    )


def get_histogram(trajectory, dx, weights=None, dt=1):
    """
    Function to generate a histogram with dx as bin size, dt is the  \
    normalization value (if applicable).
    Return:
      sbins: upper bond value for each bin;
      svalues: the number of elements inside each bin definition, divided by dx.
    """
    trajectory = np.asarray(trajectory)
    sbins = np.arange(trajectory.min(), trajectory.max(), dx)
    svalues = np.divide(
        np.asarray(
            [
                np.equal(np.digitize(trajectory[weights], sbins), x).sum()
                for x in range(1, np.shape(sbins)[0] + 1)
            ]
        ),
        dx,
    )
    return svalues, sbins


def get_state(trajectory, a, b):
    """Function to return the state for each frame, based on the following \
    criteria: below or equal a --> 0; between a and b --> 1; above or equal \
    b --> 2"""
    trajectory = np.asarray(trajectory)
    if a > b:
        a, b = b, a
    state = np.ones(shape=np.shape(trajectory)[0])
    state[np.less_equal(trajectory, a)] = 0
    state[np.greater_equal(trajectory, b)] = 2
    return state


def ret_changed_idx(state):
    """Function to return the indexes where the state has changed."""
    return np.where(np.not_equal(np.roll(state, 1), state))[0]


def get_transitions(state):
    """Function to return the transitions indexes, where the first value for \
    each row is when it starts and the second when it ends from the state \
    vector."""
    idx_changed = ret_changed_idx(state)
    checked = []
    if state[0] != 1:
        s = state[0]
    else:
        for i, j in enumerate(state):
            if j != 1:
                s = j
                break
    for i, j in enumerate(idx_changed):
        if state[j] == 2 and s != 2:
            s = 2
            checked.append([idx_changed[i - 1], j])
        if state[j] == 0 and s != 0:
            s = 0
            checked.append([idx_changed[i - 1], j])
    return np.asarray(checked)


def create_weights(trajectory, a, b):
    """
    Function to return the trajectory weights, assigning one to frames \
    involved in transitions and zeros otherwise.
    Input:
      trajectory: 1-D trajectory (numpy.array)
      a, b: transition state boundaries.
    Output:
      p(TP|x): Nx2 numpy.array
    """
    weight_transitions = np.zeros(shape=trajectory.shape[0])
    pairs_transitions = get_transitions(get_state(trajectory, a, b))
    for transition in pairs_transitions:
        weight_transitions[transition[0] : transition[1]] = 1
    return weight_transitions


def evaluate_ptpx(trajectory, a, b, nbins=50, dt=1):
    """
    Function to evaluate the probability of being in a transition path \
    given the transition boundaries.
    Input:
      trajectory: 1-D trajectory (numpy.array)
      a, b: transition state boundaries.
      nbins: number of bins to be used.
      dt: time correction (correspondent time unit of each frame).
    Output:
      p(TP|x): Nx2 numpy.array
    """
    trajectory = np.asarray(trajectory)
    # Vector with the successful transition frames tagged with one.
    weight_transitions = create_weights(trajectory, a, b)
    all_values, all_bins_edges = np.histogram(trajectory, bins=nbins)
    sel_values, sel_bins_edges = np.histogram(
        trajectory, bins=nbins, weights=weight_transitions
    )
    bins_centers = np.divide(
        np.add(all_bins_edges[1:], all_bins_edges[:-1]), 2
    )
    tvalues = np.zeros(shape=all_values.shape[0])
    fvalues = np.divide(
        sel_values, all_values, out=tvalues, where=np.not_equal(all_values, 0)
    )
    ptpx_array = np.concatenate(
        (
            np.multiply(bins_centers.reshape(-1, 1), dt),
            fvalues.reshape(-1, 1),
        ),
        axis=1,
    )
    return ptpx_array


def check_trajectory(trajectories):
    several = True
    try:
        all_trajs_concat = np.concatenate(trajectories)
    except:
        print("You gave just one trajectory.")
        all_trajs_concat = trajectories
        several = False
    return all_trajs_concat, several


def evaluate_ptpx_all(trajectories, a, b, nbins=50, dt=1):
    """
    Function to evaluate the probability of being in a transition path \
    given the transition boundaries.
    Input:
      trajectories: One or a list with several 1-D trajectories \
    (numpy.array or list).
      a, b: transition state boundaries.
      nbins: number of bins to be used.
      dt: time correction (correspondent time unit of each frame).
    Output:
      p(TP|x): Nx2 numpy.array
    """
    trajectories = np.asarray(trajectories)
    all_trajs_concat, several = check_trajectory(trajectories)
    if np.logical_not(several):
        ptpx_array = evaluate_ptpx(trajectories, a, b, nbins=nbins, dt=dt)
    else:
        all_values, all_bins_edges = np.histogram(
            all_trajs_concat, bins=nbins
        )
        sel_values_sum = np.zeros(np.shape(all_values))
        all_values_sum = np.zeros(np.shape(all_values))
        for i in trajectories:
            i = np.asarray(i)
            weight_transitions = create_weights(i, a, b)
            sel_values, _ = np.histogram(
                i, bins=all_bins_edges, weights=weight_transitions
            )
            values, _ = np.histogram(i, bins=all_bins_edges)
            sel_values_sum = np.add(sel_values_sum, sel_values)
            all_values_sum = np.add(all_values_sum, values)
        bins_centers = np.divide(
            np.add(all_bins_edges[1:], all_bins_edges[:-1]), 2
        )
        tvalues = np.zeros(shape=all_values_sum.shape[0])
        fvalues = np.divide(
            sel_values_sum,
            all_values_sum,
            out=tvalues,
            where=np.not_equal(all_values_sum, 0),
        )
        ptpx_array = np.concatenate(
            (
                np.multiply(bins_centers.reshape(-1, 1), dt),
                fvalues.reshape(-1, 1),
            ),
            axis=1,
        )
        assert np.equal(np.sum(all_values), np.sum(all_values_sum))
    return ptpx_array


def moving_average(x, w):
    """Function to evaluate the moving average from a given 1-D array
    Input:
     - x: 1D ndarray
     - w: window size
    Output:
     - ndarray with the average of size w, excluding boundaries."""
    x = np.asarray(x)
    return np.divide(np.convolve(x, np.ones(w), "valid"), w)


def extracting_equal_rows(array1, array2):
    """Function to extract the common rows between two given arrays or
    lists.
    Input:
     - array1: ndarray
     - array2: ndarray
    Output:
     - ndarray with the rows of array2 which appears on array1 also."""
    # Ensuring both inputs are ndarrays
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    # Finding where are the rows from array2 on array1
    mask = (array1[:, None] == array2).all(axis=-1).any(axis=1)
    return array1[mask]


def stack_to_virtual_dataset(
    file_names_to_concatenate, virtual_filename, entry_key="distances"
):
    """Function to stack hdf5 files into a virtual dataset. It works better if all datasets have same shape
    Input:
     - file_names_to_concatenate: sorted list of (h5 or hdf5) file names to be concatenated.
     - virtual_filename: Filename (h5 extension) of the virtual concatenated data
     - entry_key: key to be used to extract the data from every file on file_names_to_concatenate
    Output:
     - h5 file saved with stacked data."""
    sh = h5py.File(file_names_to_concatenate[0], "r")[
        entry_key
    ].shape  # get the first dataset shape.
    layout = h5py.VirtualLayout(
        shape=(len(file_names_to_concatenate),) + sh, dtype=np.double
    )
    with h5py.File(virtual_filename, "w", libver="latest") as f:
        for i, filename in enumerate(file_names_to_concatenate):
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :] = vsource
        f.create_virtual_dataset(entry_key, layout, fillvalue=0)



def concatenate_to_virtual_dataset(
    file_names_to_concatenate, virtual_filename, entry_key="distances"
):
    """Function to concatenate hdf5 files into a virtual dataset.
    Input:
     - file_names_to_concatenate: sorted list of (h5 or hdf5) file names to be concatenated.
     - virtual_filename: Filename (h5 extension) of the virtual concatenated data
     - entry_key: key to be used to extract the data from every file on file_names_to_concatenate
    Output:
     - h5 file saved with concatenated data."""
    sh = h5py.File(file_names_to_concatenate[0], "r")[
        entry_key
    ].shape  # get the first dataset shape.
    total_lines = 0
    for filename in file_names_to_concatenate:
        total_lines += h5py.File(filename, "r")[entry_key].shape[0]
    # Now fixing the layout to have the shape of all data
    layout = h5py.VirtualLayout(
        shape=(
            total_lines,
            sh[-1],
        ),
        dtype=np.double,
    )
    with h5py.File(virtual_filename, "w", libver="latest") as f:
        current_initial_position = 0
        current_final_position = 0
        for i, filename in enumerate(file_names_to_concatenate):
            cshape = h5py.File(filename, "r")[entry_key].shape
            vsource = h5py.VirtualSource(filename, entry_key, shape=cshape)
            current_final_position += h5py.File(filename, "r")[
                entry_key
            ].shape[0]
            layout[current_initial_position:current_final_position] = vsource
            current_initial_position = current_final_position
        f.create_virtual_dataset(entry_key, layout, fillvalue=0)


################################################################################
# Method imported from mdtraj package website mdtraj library                   #
################################################################################
def best_hummer_q(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajectory, or just a single frame.
        Only the first conformation is used

    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`

    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """

    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers

    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices("heavy")
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [
            (i, j)
            for (i, j) in combinations(heavy, 2)
            if abs(
                native.topology.atom(i).residue.index
                - native.topology.atom(j).residue.index
            )
            > 3
        ]
    )

    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    print("Number of native contacts", len(native_contacts))

    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)

    q = np.mean(
        1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1
    )
    return q


################################################################################


# #sequence of functions to speed up large files reading. Need to be tested.
#
# def processing_text(i, line, delimiter):
#     out[i] = line.split(delimiter)
#     return out
#
# import mmap
#
# def parallel_read_large_txt(file, eq_steps=0, delimiter=None, dtype=None):
#     #tempfile = clean_file(file)
#     with open(tempfile, "r+") as FileObj:
#         nrows = sum(1 for line in FileObj)
#         FileObj.seek(0)
#         ncols = len(next(FileObj).split(delimiter))
#         out = np.empty((nrows, ncols), dtype=dtype)
#         sout = sa.create('file://example', (nrows, ncols), dtype=dtype)
#         FileObj.seek(0)
#         print("Beatiful so far.")
#         mm = mmap.mmap(FileObj.fileno(), 0)
#         #to parallelize using N - 1 cores
#         cores = (multiprocessing.cpu_count() - 1)
#         pool = multiprocessing.Pool(cores)
#         print("We are almost there.")
#         outa = pool.starmap(processing_text, [(i, line, delimiter, sout) for \
#                                                i, line in enumerate(mm)])
#         pool.close()
#         mm.close()
#     os.remove(tempfile)
#     out = [item.get() for item in outa]
#     sout.delete('file://example')
#     return out[np.int(eq_steps):]
#
# def process_wrapper(filename, chunkStart, chunkSize):
#     with open(filename) as f:
#         f.seek(chunkStart)
#         lines = f.read(chunkSize).splitlines()
#         for line in lines:
#             out[i] = line.split(delimiter)
#
# def chunkify(fname, size=1024*1024):
#     fileEnd = os.path.getsize(fname)
#     with open(fname,'r') as f:
#         chunkEnd = f.tell()
#         while True:
#             chunkStart = chunkEnd
#             f.seek(size, 1)
#             f.readline()
#             chunkEnd = f.tell()
#             yield chunkStart, chunkEnd - chunkStart
#             if chunkEnd > fileEnd:
#                 break
#
# def aparallel_read_large_txt(file, eq_steps=0, delimiter=None, dtype=None):
#     tempfile1 = clean_file(file)
#     #init objects
#     cores = (multiprocessing.cpu_count() - 1)
#     pool = multiprocessing.Pool(cores)
#     jobs = []
#     #create jobs
#     for chunkStart, chunkSize in chunkify(tempfile1):
#         jobs.append( pool.apply(process_wrapper, (tempfile1, chunkStart, chunkSize)) )
#     #wait for all jobs to finish
#     for job in jobs:
#         job.get()
#     #clean up
#     pool.close()
#     return jobs
#
#     filename='gpdata.dat'  #your filename goes here.
# fsize=os.path.getsize(filename) #size of file (in bytes)
#
#
# #break the file into 20 chunks for processing.
# nchunks=20
# initial_chunks=range(1,fsize,fsize/nchunks)
#
# #You could also do something like:
# #initial_chunks=range(1,fsize,max_chunk_size_in_bytes) #this should work too.
#
#
# with open(filename,'r') as f:
#     start_byte=sorted(set([newlinebefore(f,i) for i in initial_chunks]))
#
# end_byte=[i-1 for i in start_byte] [1:] + [None]
#
# def process_piece(filename,start,end):
#     with open(filename,'r') as f:
#         f.seek(start+1)
#         if(end is None):
#             text=f.read()
#         else:
#             nbytes=end-start+1
#             text=f.read(nbytes)
#
#     # process text here. createing some object to be returned
#     # You could wrap text into a StringIO object if you want to be able to
#     # read from it the way you would a file.
#
#     returnobj=text
#     return returnobj
#
# def wrapper(args):
#     return process_piece(*args)
