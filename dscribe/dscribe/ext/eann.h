#ifndef EANN_H
#define EANN_H

#include <unordered_map>
#include <vector>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

using namespace std;

/**
 * Implementation for the performance-critical parts of EANN.
 */
class EANN {

    public:
        EANN() {};
        EANN(
            double rCut,
            int nwave,
            int orbital,
            vector<int> atomicNumbers,
            double alpha,
            vector<double> rs,
            vector<vector<double>> c
        );
        vector<vector<double>> create(vector<vector<double>>  &positions, \
            vector<int> &atomicNumbers, \
            const vector<vector<double>> &distances, \
            const vector<vector<int>> &neighbours, \
            vector<int> &indices );
        void setRCut(double rCut);
        void setAtomicNumbers(vector<int> atomicNumbers);
        void setnwave(int nwave);
        void setalpha(double alpha);
        void setorbital(int orbital);
        void setrs(vector<double> rs);
        void setc(vector<vector<double>> c);

        double rCut;
        int nwave;
        int orbital;
        vector<int> atomicNumbers;
        double alpha;
        vector<vector<double>> c;
        vector<double> rs;

        double getRCut();
        int getnwave();
        int getnorbital();
        double getalpha();
        vector<int> getAtomicNumbers();
        double gaussian_fuc(double &r_ij,double &alpha,double &rs);
        vector<vector<double>> getc();
        vector<double> getrs();
        double costheta(double r_ij, double r_ik, double r_jk);

    private:
        unordered_map<int, int> atomicNumberToIndexMap;

        double computeCutoff(double r_ij);
};

#endif
