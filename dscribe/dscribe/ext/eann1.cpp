#include "eann.h"
#include <tuple>
#include <map>
#include <math.h>
#include <string>
#include <numeric>
#include <utility>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <vector>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

using namespace std;

EANN::EANN(double rCut, int nwave, \
            int orbital, vector<int> atomicNumbers, double alpha, \
            vector<double> rs, \
        vector<vector<double>> c
)
{
    setRCut(rCut);
    setnwave(nwave);
    setorbital(orbital);
    setAtomicNumbers(atomicNumbers);
    setalpha(alpha);
    setrs(rs);
    setc(c);
}

void EANN::setRCut(double rCut)
{
    this->rCut = rCut;
}
double EANN::getRCut()
{
    return this->rCut;
}

void EANN::setnwave(int nwave)
{
    this->nwave = nwave;
}
int EANN::getnwave()
{
    return this->nwave;
}

void EANN::setorbital(int orbital)
{
    this->orbital = orbital;
}
int EANN::getnorbital()
{
    return this->orbital;
}

void EANN::setalpha(double alpha)
{
    this->alpha = alpha;
}
double EANN::getalpha()
{
    return this->alpha;
}

void EANN::setrs(vector<double> rs)
{
    this->rs = rs;
}
vector<double> EANN::getrs()
{
    return this->rs;
}

void EANN::setc(vector<vector<double>> c)
{
    this->c = c;
}
vector<vector<double>> EANN::getc()
{
    return this->c;
}


void EANN::setAtomicNumbers(vector<int> atomicNumbers)
{
    this->atomicNumbers = atomicNumbers;
    int nTypes = atomicNumbers.size();
    int nTypePairs = nTypes*(nTypes+1)/2;
    unordered_map<int, int> atomicNumberToIndexMap;
    int i = 0;
    for (int Z : atomicNumbers) {
        atomicNumberToIndexMap[Z] = i;
        ++i;
    }
    this->atomicNumberToIndexMap = atomicNumberToIndexMap;
}

vector<int> EANN::getAtomicNumbers()
{
    return this->atomicNumbers;
}

inline double EANN::computeCutoff(double r_ij)
{
    if (r_ij <= rCut)
    {
        return 0.5*(cos(r_ij*PI/rCut)+1);
    }
    else
    {
        return 0;
    }
}

inline double EANN::gaussian_fuc(double &r_ij,double &alpha,double &rs)
{
    return exp(-alpha*pow(r_ij-rs,2));
}

inline double EANN::costheta(double r_ij, double r_ik, double r_jk)
{
    if (r_ij <= rCut && r_ik <= rCut && r_jk <= rCut)
    {
        double r_ij_square = pow(r_ij,2);
        double r_ik_square = pow(r_ik,2);
        double r_jk_square = pow(r_jk,2);
        double costheta_i_j = 0.5/(r_ij*r_ik) * (r_ij_square+r_ik_square-r_jk_square);
        return costheta_i_j;
    }
    else
    {
        return 0;
    }
}


vector<vector<double>> EANN::create(vector<vector<double>> &positions, \
        vector<int> &atomicNumbers, \
        const vector<vector<double>> &distances, \
        const vector<vector<int>> &neighbours, \
        vector<int> &indices
        )
{
     // Allocate memory
    int nIndices = indices.size();
    int number_positions = positions.size();
    vector<vector<double> > output(nIndices,vector<double>(nwave));

    int index = 0;
    for (int &i : indices)
    {
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours)
        {
            if (i == j) {
                continue;
            }
            // Precompute some values
            double r_ij = distances[i][j];
            int index_j = atomicNumberToIndexMap[atomicNumbers[j]];
            double rs_j = rs[index_j];
            double fc_ij = computeCutoff(r_ij);
            double gaussion_out_ij = gaussian_fuc(r_ij,alpha,rs_j);
            for (const int &k : i_neighbours)
            {
                if (k == i || k >= j) {
                    continue;
                }
                // distance i, j and j, k
                double r_ik = distances[i][k];
                double r_jk = distances[j][k];
                int index_k = atomicNumberToIndexMap[atomicNumbers[k]];
                // Compute gaussian_fuc
                double rs_k = rs[index_k];
                double gaussion_out_ik = gaussian_fuc(r_ik,alpha,rs_k);
                double fc_ik = computeCutoff(r_ik);
                double costheta_i_j_k = costheta(r_ij, r_ik, r_jk);
                //cout << "gaussion i j: " << gaussion_out_ik << endl;
                //cout << "costheta i j: " << costheta_i_j << endl;
                //cout << "fc ik: " << fc_ik << endl;
                //cout << "rs k: " << gaussion_out_ik << endl;
                //cout << "fc ij: " << gaussion_out_ik << endl;
                //cout << "gaussion i k: " << gaussion_out_ik << endl;
                //cout << "gaussion out ij: " << gaussion_out_ik << endl;
                //cout << "orbital: " << orbital << endl;
                for (int n_i =0; n_i!= nwave; n_i++)
                {
                    double c_j_ni = c[index_j][n_i];
                    double c_k_ni = c[index_k][n_i];
                    double density = c_j_ni*gaussion_out_ij*fc_ij*pow(r_ij,orbital)* \
                            c_k_ni*gaussion_out_ik*fc_ik*pow(r_ik,orbital) * pow(costheta_i_j_k,orbital);
                    output[index][n_i] += density;
                }
            }
        }
        ++index;
    }
    return output;
}
