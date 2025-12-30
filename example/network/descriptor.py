# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:07:08 2024

@author: Administrator
"""

import ase.io
import dscribe.descriptors
from multiprocessing import cpu_count
import numpy as np
from dscribe.descriptors import EANN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from joblib import dump, load
from sklearn.metrics import davies_bouldin_score
import os
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel, AverageKernel
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Descriptors:
    def __init__(self, atoms, desp_type, cluster="kmeans", load_data=False, index=None, node=None):
        self.atoms = atoms
        self.desp_type = desp_type
        self.index = index
        self.load_data = load_data
        self.cluster = cluster
        if load_data == False:
            if os.path.exists("./descriptor.npy"):
                os.remove("./descriptor.npy")
        if node != 1:
            self.node = cpu_count()
        else:
            self.node = 1

    # ------------load descriptor----------------
    def load_descrptor(self, atoms=None, index=None):
        """
        out: np.array[(len(atom),index,nwave)]
        """
        if atoms == None:
            atoms = self.atoms
        desp = self.desp_type.create(atoms, index, n_jobs=self.node)
        metal_desp = desp.reshape(-1, desp.shape[-1])
        return metal_desp

    # -----------cluster-------------------------
    def cluster_train(self, n_cluster=None):
        silhouette_scores = []
        if os.path.exists("descriptor.npy"):
            self.desp = load("descriptor.npy")
        else:
            self.desp = self.load_descrptor(self.atoms, self.index)
            np.save("descriptor.npy", self.desp)
        cluster_max = min(100, int(self.desp.shape[0]/2))
        if self.cluster == "kmeans":
            if n_cluster != None:
                best_n_clusters = n_cluster
            else:
                for k in range(2, cluster_max):
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(self.desp)
                    score = silhouette_score(self.desp, kmeans.labels_)
                    silhouette_scores.append(score)

                best_n_clusters = range(
                    2, cluster_max)[silhouette_scores.index(max(silhouette_scores))]
            clustering = KMeans(n_clusters=best_n_clusters)
            clustering.fit(self.desp)
        elif self.cluster == "GMM":
            for k in range(2, cluster_max):
                gmm = GaussianMixture(n_components=k)
                gmm.fit(self.desp)
                score = silhouette_score(self.desp, gmm.fit_predict(self.desp))
                silhouette_scores.append(score)

            best_n_clusters = range(
                2, cluster_max)[silhouette_scores.index(max(silhouette_scores))]

            clustering = GaussianMixture(n_components=best_n_clusters)
            clustering.fit(self.desp)
        labels = clustering.labels_
        tsne = TSNE(n_components=2, random_state=0,
                    perplexity=min(10, cluster_max/2))
        X_tsne = tsne.fit_transform(self.desp)
        plt.figure(figsize=(8, 8))
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        colors = ['r', 'g', 'b', 'c', 'm', 'y',
                  'k', 'orange', 'purple', 'brown']
        for i in range(min(10, best_n_clusters)):
            plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1],
                        marker=markers[i], color=colors[i], alpha=0.5, label=f'Cluster {i}')
            # plt.scatter(centers_tsne[i, 0], centers_tsne[i, 1], marker=markers[i],
            #            color=colors[i], edgecolor='k', s=200, label=f'Center {i}')
        plt.legend()
        # plt.show()
        plt.savefig("2d_cluster.png")
        dump(clustering, 'clustering_model.joblib')

    # ------------test-----------------
    def test(self, atoms_test, index_test=None):
        desp_test = self.load_descrptor(atoms_test, index=index_test)
        model_ = load("clustering_model.joblib")
        y_kmeans = model_.predict(desp_test)
        db_score = davies_bouldin_score(desp_test, y_kmeans)
        print("Davies-Bouldin is :", db_score)
        return y_kmeans

    # -------------similarity------------
    def similarity(self, atoms1, atoms2, index1=None, index2=None):
        desp1 = self.load_descrptor(atoms1, index=index1)
        desp2 = self.load_descrptor(atoms2, index=index2)
        desp1 = normalize(desp1)
        desp2 = normalize(desp2)
        # Any metric supported by scikit-learn will work: e.g. a Gaussian.
        re = REMatchKernel(metric="rbf", gamma=1, alpha=1, threshold=1e-6)
        re_kernel = re.create([desp1, desp2])
        return re_kernel

    # -------------similarity all------------
    def similarity_all(self):
        desp = self.desp_type.create(
            self.atoms, self.index, n_jobs=self.node)
        re_kernel_all = np.ones((len(self.atoms), len(self.atoms)))
        re = REMatchKernel(metric="rbf", gamma=0.1,
                           alpha=1e-3, threshold=1e-6)
        for i in range(len(desp)-1):
            desp1 = desp[i]
            desp1 = normalize(desp1)
            for j in range(1, len(desp)):
                desp2 = desp[j]
                desp2 = normalize(desp2)
                # Any metric supported by scikit-learn will work: e.g. a Gaussian.
                re_kernel = re.create([desp1, desp2])
                if re_kernel[0][1] <= 1:
                    re_kernel_all[i][j] = re_kernel[0][1]
                    re_kernel_all[j][i] = re_kernel[1][0]
        return re_kernel_all
