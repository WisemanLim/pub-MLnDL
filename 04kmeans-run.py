# -*- coding: utf-8 -*-
#

import sys
import numpy as np
from scipy.cluster.vq import kmeans, vq
from numpy import genfromtxt

def readData(filename, columns = None):
    if columns != None:
        csv = genfromtxt(filename, delimiter=",",
                         usecols=(map(int, columns.split(","))))
    else:
        csv = genfromtxt(filename, delimiter=",")

    return csv

if __name__ == "__main__":
    cluster_filename = sys.argv[1]
    test_filename = sys.argv[2]
    # columns 처리 : (**)교재와 같이 무조건 컬럼없거나 또는 있을 때 수행하도록 수정
    columns = None
    if len(sys.argv) > 3: columns = sys.argv[3]

    # 모델 읽어오기 (1)
    if columns != None:
        t_list = (columns.split(","))
        c = list()
        for l in t_list:
            c.append(int(l) - 1)
        print(c)
        centroids = np.loadtxt(cluster_filename, delimiter=",",
                               usecols=c)
    else:
        centroids = np.loadtxt(cluster_filename, delimiter=",")

    read_data = readData(test_filename, columns)

    # 결과 양자화 (2)
    idx, _ = vq(read_data, centroids)
    print(idx)
    # (1)검증데이터가 wine.data에서 클러스터 정답 컬럼 제외 13개 컬럼
    # python 04kmeans-run.py wine-centroid.csv 04wine-test.data
    # (2)검증데이터가 wine.data와 동일할 때 : 클러스터 정답 컬럼 포함 14개 컬럼 (**)교재와 같이 수행시 무조건 컬럼 지정할 것
    # python 04kmeans-run.py wine-centroid.csv ./practice/chapter04/wine-test.data  "1,2,3,4,5,6,7,8,9,10,11,12,13"
    # python 04kmeans-run.py wine-centroid.csv wine-test.data  "1,2,3,4,5,6,7,8,9,10,11,12,13"
