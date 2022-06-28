import random
import time
import math
from typing import BinaryIO
import subprocess as sp
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from operator import itemgetter


def compute_distance(pointa, pointb):
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)


# make_graph
# inputs: n, the number of points to generate,
# m - the size of the [0,M]^2 area where points are pulled from
# radius - the distance between two points required for an edge to be present
# outputs: vertices, a (nx2) matrix of floats corresponding to the pointset
# edges - an (Ex2) matrix of edges (a,b) where a and b have at most radius distance
# G - a networkx graph object for the approximation algorithm.
def make_graph(n, m, radius):
    vertices = []  # prepare a list to store the 2D pointset
    for i in range(n):
        # plot all points
        vertices.append([random.uniform(0, m), random.uniform(0, m), 0])
    # here, we sort the data points in euclidean space from left to right and top to bottom
    ltr_ttb = sorted(sorted(vertices, key=itemgetter(1)), key=itemgetter(0))
    vertices = ltr_ttb
    for i in range(len(vertices)):
        # assign the number as an input
        vertices[i][2] = i
    edges = []
    for inda in range(n):
        a = vertices[inda]
        for indb in range(n):
            b = vertices[indb]
            if not a == b and compute_distance(a, b) < radius:
                edges.append((inda, indb))
                # this is an undirected, symmetric graph. both (a,b) and (b,a) end up in edges

    return (vertices, edges)
    # return as a double


# generates the encoded version of the local views
# epsilon is the size of the largest bin you want considered in the local-view
# this is the chebyshev distance from the center point, creating a 2*epsilon by 2*epsilon square
def encode(vertices, array_size = 100.0, m = 10.0):
    encoding = np.zeros((int(array_size), int(array_size)))
    bin_size = m/array_size
    for vertex in vertices:
        x_val = vertex[0]
        y_val = vertex[1]
        x_bin = int(x_val/bin_size)
        y_bin = int(y_val/bin_size)
        encoding[x_bin][y_bin]+=1
    return encoding

def encode_sols(sols, vertices, array_size = 100.0, m = 10.0):
    encoding = np.zeros((int(array_size), int(array_size)))
    bin_size = m / array_size
    for sol in sols:
        x_val = vertices[sol][0]
        y_val = vertices[sol][1]
        x_bin = int(x_val/bin_size)
        y_bin = int(y_val/bin_size)
        encoding[x_bin][y_bin]+=1
    return encoding


def get_mis_approx(v, e, num_process=0):
    # identify the num_process to produce a unique file name for the approximator
    n_p = str(num_process)
    set = []
    num_vertices = len(v)
    # number of undirected edges is half the number of total directed edges
    num_edges = int(len(e) / 2.0)
    # begin creating the graph file for work by the KaMIS solver (layout on documentation)
    string_to_write = str(str(num_vertices) + " " + str(num_edges) + '\n')
    edges = {}
    # create an edge dictionary to speed creation of file
    for vertex in range(1, num_vertices + 1):
        edges[vertex] = []
    for (e1, e2) in e:
        edges[e1 + 1].append(e2 + 1)
    # each line corresponds to all the outgoing edges from that vertex (1-indexed, hence the +1 in the ln above)
    for vertex in range(1, num_vertices + 1):
        out_edges = sorted(edges[vertex])
        line = ""
        for elem in out_edges:
            line += str(elem) + " "
        line += '\n'
        # print(line)
        string_to_write += line
    # compile the string in memory, then write to disk at one time to minimize file I/O
    f = open("./graph_temp_" + str(num_vertices) + "_" + n_p + ".txt", 'w')
    f.write(string_to_write)
    f.close()
    # call the KaMIS solver and wait until it completes
    sp.call(["/KaMIS-2.0/deploy/redumis",
             "./graph_temp_" + str(
                 num_vertices) + "_" + n_p + ".txt",
             "--output=./output/output_" + str(
                 num_vertices) + "_" + n_p + ".out"], stdout=sp.DEVNULL)
    # print an indicator to show work progress/potential race conditions
    print(num_vertices, num_edges, num_process)
    # read the output from the
    o = open("./output/output_" + str(num_vertices) + "_" + n_p + ".out", "r")
    i = 0
    for line in o:
        if '1' in line:
            set.append(i)
        i += 1
    o.close()
    # print(set)
    return set


# this is the function that is parallelized to produce the total dataset
# m = size of total space space, r = 2*radius, approx = if True, use a 2*opt approximation alg
def write_data(n_lo, n_hi, m, r, to_encode=True, to_approx=True, array_size=100.0, num_process=0):
    n = random.randrange(n_lo, n_hi)
    v, e = make_graph(n, m, r)
    if not to_approx:
        exit(1)
    else:
        set = get_mis_approx(v, e, num_process)
    if not to_encode:
        return [m, r, [], set, v, e, False, to_approx]
    else:
        return [m, r, encode(v, array_size,m), encode_sols(set,v,array_size,m), v, e, array_size, set]

    # print((n,len(cover),memo,elapsed))


def dataset_construction(n_lo=400, n_hi=500, m=10.0, r=2.0,
                         to_encode=True, approx=True, iteration=0, num_iter=1,
                         array_size = 100.0, to_para=False):
    eps = float(eps)

    # time the parallel process
    strt = time.time()
    if to_para:
        with Parallel(n_jobs=num_iter) as para:
            toWrite = []
            # each dataset will get its own process while it solves the solution
            dataset = para(
                delayed(write_data)(n_lo, n_hi, m, r, to_encode, approx, array_size, i) for i in range(num_iter))
            # add the triple to the document which will be written to disk
            toWrite.extend(dataset)
    if not to_para:
        toWrite = []
        # each dataset will get its own process while it solves the solution
        for i in range(num_iter):
            startindiv = time.time()
            dataset = write_data(n_lo, n_hi, m, r, to_encode, approx, array_size, iteration)
            # add the triple to the document which will be written to disk
            toWrite.append(dataset)
            print(iteration, time.time() - startindiv)


    # convert to nparray to save to disk
    toSave = np.array(toWrite)
    # indicate in the file name if the data set is exact or approximate
    filename = "./data.npy"

    # CAN BE CLEANED: tried to write appending file writer that does not erase work already done.
    p = Path(filename)
    with p.open('wb+') as f:
        f: BinaryIO
        np.save(f, toSave)
        f.close()

    tm = time.time() - strt
    # print progress and running time
    print(iteration, tm)



def parallel_dc(num_tasks=16, num_files=10, n_lo=400,
                n_hi=500, m=10.0, r=2.0,
                to_encode=True, approx=True, iteration=0, num_iter=1,
                array_size=100.0, to_para=False):
    with Parallel(n_jobs=num_tasks) as para:
        para(delayed(dataset_construction)(n_lo=n_lo,n_hi=n_hi,m=m,r=r,to_encode=to_encode,
                                           approx=approx,iteration=i,
                                           num_iter=num_iter,array_size=array_size,to_para=to_para) for i in range(num_files))

