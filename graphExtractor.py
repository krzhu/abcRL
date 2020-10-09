##
# @file graphExtractor.py
# @author Keren Zhu
# @date 11/16/2019
# @brief The functions and classes for processing the graph
#

import abc_py as abcPy
import numpy as np
from numpy import linalg as LA
import numpy as np
import dgl
import torch

def symmetricLaplacian(abc):
    numNodes = abc.numNodes()
    L = np.zeros((numNodes, numNodes))
    print("numNodes", numNodes)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        degree = float(aigNode.numFanouts())
        if (aigNode.hasFanin0()):
            degree += 1.0
            fanin = aigNode.fanin0()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        if (aigNode.hasFanin1()):
            degree += 1.0
            fanin = aigNode.fanin1()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        L[nodeIdx][nodeIdx] = degree
    return L

def symmetricLapalacianEigenValues(abc):
    L = symmetricLaplacian(abc)
    print("L", L)
    eigVals = np.real(LA.eigvals(L))
    print("eigVals", eigVals)
    return eigVals

def extract_dgl_graph(abc):
    numNodes = abc.numNodes()
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 6)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        features[nodeIdx][nodeType] = 1.0
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            G.add_edge(fanin, nodeIdx)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            G.add_edge(fanin, nodeIdx)
    G.ndata['feat'] = torch.tensor(features)
    return G

