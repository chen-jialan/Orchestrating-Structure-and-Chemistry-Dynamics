# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:22:49 2024

@author: lenovo
"""

import numpy as np
import math
from functools import reduce
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
import copy
import matplotlib.pyplot as plt
import scienceplots
from pandas import DataFrame
import time
import os
import shutil
import sys
import time
# import extensisq


class MicroKinetic:
    """
    Method is for solve odes, default is Radau
    Theory of microkinetic modeling is based on collision theory and transition state theory
    We will calculate the TOF and selection of production,
    """

    def __init__(self, species, coverage, element_step, reaction_species, product_species,
                 temperature_start=300, temperature_final=1000, temperature_point=10, pressure=1e5,
                 t_max=1e5, method="BDF", atol=1e-6, rtol=1e-6, dt=1000, ssia_calc=True,
                 coverage_run=True, order_run=True, drc_run=True, apparent_energy_run=True,
                 element_step_adjust=None,
                 # boundary_limit=None,
                 filename="mkm"):
        """
        there are several paremeters of inputs:

        @param list -> species              species of reaction pathway
        @param list -> coverage             coverage of species in reaction pathway
        @param list[list] -> element_step   element steps of reaction pathway
        @param list -> reaction_species     reaction species
        @param list -> product_species      product species
        @param float -> temperature_start   start temperature(default 300)
        @param float -> temperature_final   final temperature(default 500)
        @param int -> temperature_point     split temperature from start temperature to final temperature(default 10)
        @param float -> pressure            pressure of external enviroment(default 1e5)
        @param int -> t_max                 time when odes solves(default 50)
        @param str -> method                odes algirithm(default BDF)
        @param bool-> coverage_run       calculate coverage
        @param bool -> order_run          calculate order
        @param bool ->  drc_run           calculate drc
        @param bool -> apparent_energy_run calculate apparent energy
        @param list[list] -> element_step_adjust  calculate adjust energy with different coverage
        """
        self.element_step = element_step
        self.species = species
        self.reaction_species = reaction_species
        self.product_species = product_species
        self.index_reaction_species = [species.index(
            spe_e) for spe_e in self.reaction_species]
        self.index_product_species = [species.index(
            spe_e) for spe_e in self.product_species]
        self.coverage = np.array(coverage, dtype=np.double)
        self.temperature_range = (np.linspace(
            temperature_start, temperature_final, temperature_point)).tolist()
        self.kB = 8.615E-5
        self.t_max = t_max
        self.ssia_calc = ssia_calc
        self.dt = dt
        self.Temp = 300
        self.pressure = pressure
        self.standard_pressure = 1e5
        self.method = method
        self.R = 8.314
        self.N = 6.02*1E23
        self.h = 6.626*1E-34
        self.mass_au = 1.674*1E-27
        self.atol = atol
        self.rtol = rtol
        self.str2reaction_all = self.str2reaction()
        self.coverage_run = coverage_run
        self.order_run = order_run
        self.drc_run = drc_run
        self.apparent_energy_run = apparent_energy_run
        self.element_step_adjust = element_step_adjust
        # if element_step_adjust != None:

        # self.boundary_limit = boundary_limit
        self.time_start = time.strftime(
            '%b_%d_%Y_%H_%M_%S', time.localtime(time.time()))
        self.filename = filename+"_"+self.time_start
        if os.path.exists("./%s" % self.filename):
            shutil.rmtree("./%s" % self.filename)
        os.mkdir("./%s" % self.filename)
        self.origin_ode = ["RK45", "RK23", "DOP853", "Radau",
                           "BDF", "LSODA"]

    def odes_functions(self, t, initial_state):
        """
        odes functions with elementary step are matrix
        coverage_m is the matrix
        k_Ea_bolz is k*exp(-Ea/kB*temperature)
        @param  list -> initial_state   initial state of odes
        @param  float -> t              odes solution
        @return numpy -> dAdt           odes of every species
        """
        dAdt = self.elementeny_equation_all(
            initial_state=initial_state, solve=True)
        return dAdt

    def elementeny_equation_all(self, initial_state=None, solve=True, k_change=None):
        """
        calculation all elementary rate and odes
        @param  list -> initial_state   initial state of odes
        @return numpy -> dAdt           odes of every species
        """
        # print(initial_state)
        dAdt = np.zeros(initial_state.shape)

        self.calc_k(k_change=k_change, initial_state=initial_state)
        if solve == True:
            for index in self.index_reaction_species:
                initial_state[index] = self.coverage[index]
            for index in self.index_product_species:
                initial_state[index] = self.coverage[index]
        coverage_m_forward = self.coverage_matrix(initial_state, forward=True)
        coverage_prod_forward = np.prod(coverage_m_forward, axis=1)
        rate_forward = coverage_prod_forward * self.k_Ea_bolz_forward

        # backward pathway
        coverage_m_backward = self.coverage_matrix(
            initial_state, forward=False)
        coverage_prod_backward = np.prod(coverage_m_backward, axis=1)
        rate_backward = coverage_prod_backward * self.k_Ea_bolz_backward

        # if initial_state[4] >= 0.5:
        #     time.sleep(1)
        #     # print(coverage_m_backward)
        #     print(initial_state)
        #     print(rate_forward)
        #     print(rate_backward)
        #     print(self.k_Ea_bolz_forward[0], self.k_Ea_bolz_backward[0])

        for i in range(len(self.element_step)):
            # forward pathway
            for j in range(len(self.element_step[i][0])):
                index = self.species.index(self.element_step[i][0][j])
                dAdt[index] += (-rate_forward[i] + rate_backward[i])
            # backward pathway
            for j in range(len(self.element_step[i][1])):
                index = self.species.index(self.element_step[i][1][j])
                dAdt[index] += (rate_forward[i] - rate_backward[i])
        # if initial_state[4] >= 0.5:
        #     print(dAdt)
        if solve == True:
            for index in self.index_reaction_species:
                dAdt[index] = 0
            for index in self.index_product_species:
                dAdt[index] = 0
        return dAdt

    def equation_species(self, y, initial_state):
        for index in self.index_reaction_species:
            initial_state[index] = 0
        for index in self.index_product_species:
            initial_state[index] = 0
        return sum(initial_state)-1

    def coverage_matrix(self, initial_state, forward=True):
        """
        coverage_matrix for elementary
        @param  list -> initial_state   initial state of odes
        @param  bool -> forward         whether or not forward reaction step
        @return numpy -> coverage       coverage of every species in elementary step
        """
        coverage_m = np.ones((len(self.element_step), len(self.species)))
        if forward == True:
            for i in range(len(self.element_step)):
                for j in range(len(self.element_step[i][0])):
                    index = self.species.index(self.element_step[i][0][j])
                    coverage_m[i][index] *= initial_state[index]
        else:
            for i in range(len(self.element_step)):
                for j in range(len(self.element_step[i][1])):
                    index = self.species.index(self.element_step[i][1][j])
                    coverage_m[i][index] *= initial_state[index]
        return coverage_m

    def elementary_ea(self, temperature, initial_state=None, forward=True):
        """
        solve the function of k*exp(-Ea/kB*T) for all elementary
        (a) with the adsorption of molecule, we use Collision Theory
        k_a = k*P_a/sqrt(2*pi*ma*kb*T)
        k_des = (kb*T**3/h**3) * A(2*pi*ma*kb*T)/(sigma*theta)exp(-Edes/kbT)
        (b) with the elementary of exp(-Ea/kb*T)
        """
        k_Ea_bolz = np.zeros(len(self.element_step))
        # print(f"initial_state is {initial_state}")

        if self.element_step_adjust == None:
            for i in range(len(self.element_step)):
                if len(self.element_step[i]) == 6:
                    if forward == True:
                        k_Ea_bolz[i] = self.element_step[i][2] * \
                            np.exp(-self.element_step[i]
                                   [4]/(self.kB*temperature))
                    else:
                        k_Ea_bolz[i] = self.element_step[i][3] * \
                            np.exp(-self.element_step[i]
                                   [5]/(self.kB*temperature))
                else:
                    mass = self.element_step[i][3]*self.mass_au
                    k_b = self.R/self.N
                    A = self.element_step[i][2][0]
                    sigma = self.element_step[i][2][1]
                    theta_rot = self.element_step[i][2][2]
                    if forward == True:
                        k_Ea_bolz[i] = A*self.pressure / \
                            np.sqrt(2*math.pi*mass*k_b*self.Temp)
                    else:
                        k_Ea_bolz[i] = k_b * math.pow(temperature, 3)/math.pow(self.h, 3) * A*(2*math.pi*mass*k_b) / \
                            (sigma*theta_rot) * \
                            np.exp(-self.element_step[i]
                                   [-1]/(self.kB*temperature))
        else:
            for i in range(len(self.element_step)):
                if len(self.element_step[i]) == 6:
                    if forward == True:
                        k_Ea_bolz[i] = self.element_step[i][2] * \
                            np.exp(-self.element_step[i]
                                   [4]/(self.kB*temperature))
                    else:
                        k_Ea_bolz[i] = self.element_step[i][3] * \
                            np.exp(-self.element_step[i]
                                   [5]/(self.kB*temperature))
                    for j in range(len(self.element_step_adjust)):
                        number_element_step = self.element_step_adjust[j][0]
                        if number_element_step == i:
                            limit_para = len(self.element_step_adjust[j][1])
                            coverage_para_all = []
                            range_all = []
                            for k in range(limit_para):
                                coverage_para = self.element_step_adjust[j][1][k][0]
                                coverage_para_all.append(coverage_para)
                                range_k = self.element_step_adjust[j][1][k][1]
                                range_all.append(range_k)
                            for k in range(len(coverage_para_all)):
                                coverage_k = initial_state[coverage_para_all[k]]
                                range_k = range_all[k]
                                check_ = self.check_ranges(coverage_k, range_k)
                                # print(check_ != -1,len(self.element_step_adjust[j]))

                                if check_ != -1 and len(self.element_step_adjust[j]) == 5 and forward == self.element_step_adjust[j][-1]:
                                    k_Ea_bolz[i] = 0
                                    for l in range(len(self.element_step_adjust[j][2])):
                                        barrier_energy_l = self.element_step_adjust[j][2][l]
                                        probablity_l = self.element_step_adjust[j][3][l]
                                        # print("--------------start adjust energy-------------------")
                                        k_Ea_bolz[i] += probablity_l * self.element_step[i][2] * \
                                            np.exp(-barrier_energy_l /
                                                   (self.kB*temperature))
                else:
                    mass = self.element_step[i][3]*self.mass_au
                    k_b = self.R/self.N
                    A = self.element_step[i][2][0]
                    sigma = self.element_step[i][2][1]
                    theta_rot = self.element_step[i][2][2]
                    if forward == True:
                        k_Ea_bolz[i] = A*self.pressure / \
                            np.sqrt(2*math.pi*mass*k_b*self.Temp)
                    else:
                        k_Ea_bolz[i] = k_b * math.pow(temperature, 3)/math.pow(self.h, 3) * A*(2*math.pi*mass*k_b) / \
                            (sigma*theta_rot) * \
                            np.exp(-self.element_step[i]
                                   [-1]/(self.kB*temperature))
                        if len(self.element_step[i]) == 4:
                            for j in range(len(self.element_step_adjust)):
                                number_element_step = self.element_step_adjust[j][0]
                                if number_element_step == i:
                                    limit_para = len(
                                        self.element_step_adjust[j][1])
                                    coverage_para_all = []
                                    range_all = []
                                    for k in range(limit_para):
                                        coverage_para = self.element_step_adjust[j][1][k][0]
                                        coverage_para_all.append(coverage_para)
                                        range_k = self.element_step_adjust[j][1][k][1]
                                        range_all.append(range_k)
                                    for k in range(len(coverage_para_all)):
                                        coverage_k = initial_state[coverage_para_all[k]]
                                        range_k = range_all[k]
                                        check_ = self.check_ranges(
                                            coverage_k, range_k)
                                        if check_ != -1 and len(self.element_step_adjust[j]) == 4:
                                            k_Ea_bolz[i] = 0
                                            for l in range(len(self.element_step_adjust[j][2])):
                                                barrier_energy_l = self.element_step_adjust[j][2][l]
                                                probablity_l = self.element_step_adjust[j][3][l]
                                                k_Ea_bolz[i] += probablity_l*k_b * math.pow(temperature, 3)/math.pow(self.h, 3) * \
                                                    A*(2*math.pi*mass*k_b) / \
                                                    (sigma*theta_rot) * \
                                                    np.exp(-barrier_energy_l /
                                                           (self.kB*temperature))
                                        else:
                                            k_Ea_bolz[i] = k_b * math.pow(temperature, 3)/math.pow(self.h, 3) * A * (2*math.pi*mass*k_b) / \
                                                (sigma*theta_rot) * \
                                                np.exp(-self.element_step[i]
                                                       [-1]/(self.kB*temperature))
                        else:
                            k_Ea_bolz[i] = k_b * math.pow(temperature, 3)/math.pow(self.h, 3) * A*(2*math.pi*mass*k_b) / \
                                (sigma*theta_rot) * \
                                np.exp(-self.element_step[i]
                                       [-1]/(self.kB*temperature))
        return k_Ea_bolz

    def check_ranges(self, num, ranges):
        """
        checke number whether or not in the ranges
        @param  float -> num   coverage
        @param  tuple -> ranges the range
        @return int -> result (-1 is False)
        """
        if ranges[0] <= num <= ranges[1]:
            return 0
        return -1

    def odes_solution(self, k_change=None):
        """
        odes solution with every elementary step by scipy package
        @return numpy -> result     every step of coverage
        """
        # t = np.arange(0.0, 1e3, 0.1)

        all_result = []

        fontsize_title = 10
        fontsize_txt = 8
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

        for Temp in self.temperature_range:
            fig, ax = plt.subplots(figsize=(4, 3))
            self.Temp = Temp
            coverage_copy = copy.deepcopy(self.coverage)
            # ------------ k -----------------
            # self.calc_k(k_change)
            solution = solve_ivp(self.odes_functions,
                                 t_span=(0, self.t_max),
                                 y0=coverage_copy,
                                 # t_eval=t,
                                 # max_steps=0.01,
                                 # event=self.equation_species,
                                 # method=extensisq.Me4,
                                 method=self.method,
                                 # dense_output=True,
                                 atol=self.atol,
                                 rtol=self.rtol)
            if solution.success:
                print("odes: %.4f K temperature has finished" % Temp)
                all_result.append(solution.y[:, :])
            else:
                if self.ssia_calc == False:
                    print("odes: %.4f K temperature may be not converaged" % Temp)
                    all_result.append(solution.y[:, :])
                else:
                    t_eval, solution = self.ssia_solver(coverage_copy)
                    print("newton and odes: %.4f K temperature has finished" % Temp)
                    solution = solution.T
                    all_result.append(solution)

            # plot coverage
            if not os.path.exists("./%s/%s" % (self.filename, Temp)):
                os.mkdir("./%s/%s" % (self.filename, Temp))

                fontsize_title = 12
                plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
                plt.rcParams['font.family'] = "Arial"
                times = []
                coverage_inter = []
                name_all = []
                for i in range(len(self.species)):
                    if i not in self.index_reaction_species and i not in self.index_product_species:
                        name = self.species[i]
                        name_all.append(name)
                        coverage_i = all_result[-1][i]
                        coverage_inter.append(coverage_i)
                        # x = np.array([i*self.t_max/len(coverage_i)
                        #             for i in range(len(coverage_i))])
                        x = solution.t[:]
                        x[0] = x[1]*0.1
                        x = np.log10(x)
                        times = x
                        plt.plot(x, coverage_i, '--', label=name)

                nocl = math.ceil(len(self.species) / 6)
                plt.legend(ncol=nocl, frameon=False)
                plt.xlabel('Times (log(s))', fontsize=fontsize_title)
                plt.ylabel('Coverage', fontsize=fontsize_title)
                # -----------save------------------
                fig.savefig('./%s/%s/coverage.eps' % (self.filename, Temp),
                            format='eps',
                            bbox_inches='tight', dpi=300)
                fig.savefig('./%s/%s/coverage.tif' % (self.filename, Temp),
                            format='tif',
                            bbox_inches='tight', dpi=300)
                keys = ["Times"]
                keys.extend(name_all)
                value = copy.deepcopy([times])
                value.extend(coverage_inter)
                dict_coverage = dict(zip(keys, value))
                df = DataFrame(dict_coverage)
                df.to_csv('./%s/%s/coverage.csv' % (self.filename, Temp))

        return all_result

    # coverage sensitivity

    def coverage_sensitivity(self, y):
        sensitivity = np.max(np.abs(y))
        return sensitivity

    # newton solution
    def damped_newton_optimization(self, func, y_init, max_iter=50):
        y = y_init
        for _ in range(max_iter):
            jacobian = np.eye(len(y))
            f_val = func(y)
            delta = np.linalg.solve(jacobian, f_val)
            parameter = 1e-1
            while np.max(delta*parameter) > 0.01:
                parameter /= 2
            y -= delta*parameter
            if np.linalg.norm(delta) < self.atol:
                break
        # else:
        #    raise RuntimeError("Newton's method did not converge")
        return y

    # SSIA solution
    def ssia_solver(self, y0):
        t_span = (0, self.t_max/100)

        t_eval = np.arange(t_span[0], t_span[1] + self.dt, self.dt)
        solution = np.zeros((len(t_eval), len(y0)))
        solution[0, :] = y0
        for i in range(1, len(t_eval)):
            t_n = t_eval[i - 1]
            y_n = solution[i - 1, :]
            low_precision_sol = solve_ivp(self.odes_functions,
                                          t_span=(t_n, t_n + self.dt),
                                          y0=y_n,
                                          method=self.method,
                                          atol=self.atol,
                                          rtol=self.rtol).y[:, -1]
            # print(low_precision_sol)
            # print(self.odes_functions(t_n + 1e-5,low_precision_sol))

            def func(y): return y - y_n - self.dt * \
                self.odes_functions(t_n + self.dt, y)
            high_precision_sol = self.damped_newton_optimization(
                func, low_precision_sol, 1000)
            solution[i, :] = high_precision_sol
            sensitivity = np.max(np.abs(high_precision_sol))
            print("%s have finished with newton solution" % t_n)
            # high_precision_sol = solve_ivp(self.odes_functions,
            #                 t_span=(t_n, t_n + self.dt),
            #                 y0=high_precision_sol,
            #                 method=self.method,
            #                 atol=self.atol,
            #                 rtol=self.rtol).y[:,-1]
#
            solution[i, :] = high_precision_sol
            # print(solution[i, :])
            if sensitivity <= 1:
                break

        return t_eval, solution

    def calc_k(self, k_change=None, initial_state=None):
        # ------------ k -----------------
        if type(k_change) is np.ndarray:
            self.k_Ea_bolz_forward = self.elementary_ea(
                temperature=self.Temp, initial_state=initial_state, forward=True)*k_change

            self.k_Ea_bolz_backward = self.elementary_ea(
                temperature=self.Temp, initial_state=initial_state, forward=False)*k_change
        else:
            self.k_Ea_bolz_forward = self.elementary_ea(
                temperature=self.Temp, initial_state=initial_state, forward=True)

            self.k_Ea_bolz_backward = self.elementary_ea(
                temperature=self.Temp, initial_state=initial_state, forward=False)

    def ode_test(self):
        self.k_Ea_bolz_forward = self.elementary_ea(
            temperature=self.Temp, forward=True)
        self.k_Ea_bolz_backward = self.elementary_ea(
            temperature=self.Temp, forward=False)
        coverage_copy = copy.deepcopy(self.coverage)
        rate_all = self.elementeny_equation_all(
            initial_state=coverage_copy)
        # print(rate_all)

    def post_analysis(self):
        all_result = self.odes_solution()

        self.plot_reaction_product(all_result)
        if self.coverage_run:
            self.plot_coverage(all_result)
        if self.apparent_energy_run:
            self.plot_apparent_energy(all_result)
        if self.order_run:
            self.plot_order(all_result)
        if self.drc_run:
            self.degree_rate_control(all_result)

        shutil.copy("%s" % sys.argv[0], "./%s/input.py" % self.filename)
        shutil.copy("./microkinetic.py", "./%s/microkinetic.py" %
                    self.filename)

    def plot_reaction_product(self, all_result):
        """
        plot of reaction and production
        @param list[numpy] -> all_result      all coverage data
        """
        print("#---------- start to calculate rate of reaction and product ----------")
        plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
        plt.rcParams['font.family'] = "Arial"
        fig, ax = plt.subplots(figsize=(4, 3))

        fontsize_title = 10
        fontsize_txt = 8
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

        index_react = self.index_reaction_species
        index_pro = self.index_product_species
        index_all = copy.deepcopy(index_react)
        index_all.extend(index_pro)
        name_all = copy.deepcopy(self.reaction_species)
        name_all.extend(self.product_species)
        r_all = [[] for i in range(len(index_all))]

        for i in range(len(self.temperature_range)):
            self.Temp = self.temperature_range[i]
            # ------------ k -----------------
            self.calc_k(initial_state=all_result[i][:, -1])
            rate_all = self.elementeny_equation_all(
                initial_state=all_result[i][:, -1], solve=False)
            for j in range(len(index_all)):
                index_j = index_all[j]
                r_all[j].append(rate_all[index_j])
        keys = ["Temperature"]
        keys.extend(name_all)
        value = copy.deepcopy([self.temperature_range])
        value.extend(r_all)
        # print(keys,value)
        dict_rate = dict(zip(keys, value))
        # print(dict_rate)
        df = DataFrame(dict_rate)
        df.to_excel('./%s/rate.xlsx' % self.filename)
        df.to_csv('./%s/rate.csv' % self.filename)

        # ---- plot -------
        for i in range(len(r_all)):
            r = r_all[i]
            name = self.species[index_all[i]]
            plt.plot(self.temperature_range, r, '--', label=name)
            # plt.grid(linestyle='--')
        nocl = math.ceil(len(name_all) / 6)
        plt.legend(ncol=nocl, frameon=False)
        plt.xlabel('Temperature (K)', fontsize=fontsize_title)
        plt.ylabel('TOF (s$^{-1}$)', fontsize=fontsize_title)
        # plt.show()

        # -----------save------------------
        fig.savefig('./%s/reaction_product.eps' % self.filename,
                    format='eps',
                    bbox_inches='tight', dpi=300)
        fig.savefig('./%s/reaction_product.tif' % self.filename,
                    format='tif',
                    bbox_inches='tight', dpi=300)

    def plot_coverage(self, all_result):
        """
        plot of coverage of species
        @param list[numpy] -> all_result      all coverage data
        """
        # name = self.species
        print("#---------- start to calculate coverage of intermediates ----------")
        plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
        plt.rcParams['font.family'] = "Arial"
        fig, ax = plt.subplots(figsize=(4, 3))
        fontsize_title = 10
        fontsize_txt = 8
        coverage_all = [[] for i in range(len(self.species))]
        for i in range(len(self.temperature_range)):
            for j in range(len(self.species)):
                coverage_all[j].append(all_result[i][j, -1])

        name_all = []
        coverage_inter = []
        for i in range(len(self.species)):
            if i not in self.index_reaction_species and i not in self.index_product_species:
                name = self.species[i]
                name_all.append(name)
                coverage_i = coverage_all[i]
                coverage_inter.append(coverage_all[i])
                plt.plot(self.temperature_range,
                         coverage_i, '--', label=name)

        nocl = math.ceil(len(self.species) / 6)
        plt.legend(ncol=nocl, frameon=False)
        plt.xlabel('Temperature (K)', fontsize=fontsize_title)
        plt.ylabel('Coverage', fontsize=fontsize_title)
        # plt.show()

        keys = ["Temperature"]
        keys.extend(name_all)
        value = copy.deepcopy([self.temperature_range])
        value.extend(coverage_inter)
        # print(keys,value)
        dict_coverage = dict(zip(keys, value))
        # print(dict_rate)
        df = DataFrame(dict_coverage)
        df.to_excel('./%s/coverage.xlsx' % self.filename)
        df.to_csv('./%s/coverage.csv' % self.filename)

        # -----------save------------------
        fig.savefig('./%s/coverage.eps' % self.filename,
                    format='eps',
                    bbox_inches='tight', dpi=300)
        fig.savefig('./%s/coverage.tif' % self.filename,
                    format='tif',
                    bbox_inches='tight', dpi=300)

    def plot_order(self, all_result):
        """
        plot of order of species
        @param list[numpy] -> all_result      all coverage data
        """
        print("#---------- start to calculate orders ----------")
        all_result_ini = all_result
        diff = [0.99, 1.01]
        order_all = [[] for i in range(len(self.product_species))]
        coverage_copy = copy.deepcopy(self.coverage)

        # all_result_diff = [[] for j in range(len(self.index_reaction_species))]
        fig, ax = plt.subplots(figsize=(4, 3))
        r_all = [[] for i in range(len(self.product_species))]
        for i in range(len(self.temperature_range)):
            self.Temp = self.temperature_range[i]
            # ------------ k -----------------
            self.calc_k(initial_state=all_result[i][:, -1])
            rate_all = self.elementeny_equation_all(
                initial_state=all_result_ini[i][:, -1], solve=False)
            for j in range(len(self.index_product_species)):
                index_j = self.index_product_species[j]
                r_all[j].append(rate_all[index_j])

        rate_all_pressure = [[[r_all[j]]
                              for i in range(len(self.index_reaction_species))]
                             for j in range(len(self.index_product_species))]
        for d_i in range(len(diff)):
            diff_i = diff[d_i]

            for j in range(len(self.index_reaction_species)):
                self.coverage = copy.deepcopy(coverage_copy)
                self.coverage[self.index_reaction_species[j]] *= diff_i
                # print(self.coverage)
                r_all_e = [[] for i in range(len(self.product_species))]
                all_result_ini = self.odes_solution()
                for t in range(len(self.temperature_range)):
                    self.Temp = self.temperature_range[t]
                    # ------------ k -----------------
                    self.calc_k(initial_state=all_result[i][:, -1])
                    rate_all = self.elementeny_equation_all(
                        initial_state=all_result_ini[t][:, -1], solve=False)
                    # print(rate_all)
                    for l in range(len(self.index_product_species)):
                        index_l = self.index_product_species[l]
                        r_all_e[l].append(rate_all[index_l])
                for k in range(len(self.index_product_species)):
                    rate_all_pressure[k][j].append(r_all_e[k])

        self.coverage = coverage_copy
        for i in range(len(rate_all_pressure)):
            for j in range(len(rate_all_pressure[i])):
                diff0_r = np.log(np.array(rate_all_pressure[i][j][0]))
                diff1_r = np.log(np.array(rate_all_pressure[i][j][1]))
                diff2_r = np.log(np.array(rate_all_pressure[i][j][2]))
                diff_r = ((diff0_r-diff1_r) /
                          (1-diff[0])+(diff2_r-diff0_r)/(diff[1]-1))/2
                order_all[i].append(diff_r)

        plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
        plt.rcParams['font.family'] = "Arial"
        fontsize_title = 10
        fontsize_txt = 8
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        name_all = self.reaction_species
        for i in range(len(self.index_product_species)):
            for j in range(len(self.index_reaction_species)):
                plt.plot(self.temperature_range,
                         order_all[i][j], '--', label=name_all[j])

        for i in range(len(self.product_species)):
            keys = ["Temperature"]
            keys.extend(name_all)
            value = copy.deepcopy([self.temperature_range])
            value.extend(order_all[i])
            dict_order = dict(zip(keys, value))
            df = DataFrame(dict_order)
            df.to_excel('./%s/order_%s.xlsx' %
                        (self.filename, self.product_species[i]))
            df.to_csv('./%s/order_%s.csv' %
                      (self.filename, self.product_species[i]))

        # max_y = [for i in range(len(self.index_product_species))]
        # y_locator = round(max()/5)
        # y_major_locator = plt.MultipleLocator(0.6)
        # ax.yaxis.set_major_locator(y_major_locator)
        nocl = math.ceil(len(name_all) / 6)
        plt.legend(ncol=nocl, frameon=False)
        plt.xlabel('Temperature (K)', fontsize=fontsize_title)
        plt.ylabel('Reaction order', fontsize=fontsize_title)
        # plt.show()
        # -----------save------------------
        fig.savefig('./%s/order.eps' % self.filename,
                    format='eps',
                    bbox_inches='tight', dpi=300)
        fig.savefig('./%s/order.tif' % self.filename,
                    format='tif',
                    bbox_inches='tight', dpi=300)

    def plot_apparent_energy(self, all_result):
        """
        plot of apparent energe
        @param list[numpy] -> all_result      all coverage data
        """
        print("#---------- start to calculate apparent energy ----------")
        plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
        plt.rcParams['font.family'] = "Arial"
        fig, ax = plt.subplots(figsize=(4, 3))
        fontsize_title = 10
        fontsize_txt = 8
        apparent_energy = [[] for i in range(len(self.index_product_species))]
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

        # print(len(self.temperature_range))
        for i in range(len(self.temperature_range)):
            for j in range(len(self.index_product_species)):
                index_pro = self.index_product_species[j]
                if i == 0 or i == len(self.temperature_range) - 1:
                    if i == 0:
                        t0 = i
                        t1 = i+1
                    else:
                        t0 = i-1
                        t1 = i
                    # -------------------r0 ----------------
                    temp = self.temperature_range[t0]
                    self.Temp = self.temperature_range[t0]
                    # self.calc_k(initial_state=all_result[i][:, -1])
                    rate_all = self.elementeny_equation_all(
                        initial_state=all_result[t0][:, -1], solve=False)
                    rate0 = rate_all[index_pro]
                    # -------------------r1 ----------------
                    temp = self.temperature_range[t1]
                    self.Temp = self.temperature_range[t1]
                    # self.calc_k(initial_state=all_result[i][:, -1])
                    rate_all = self.elementeny_equation_all(
                        initial_state=all_result[t1][:, -1], solve=False)
                    rate1 = rate_all[index_pro]
                    derivative = (math.log(rate1, math.e)-math.log(rate0, math.e)) / \
                        (self.temperature_range[t1]-self.temperature_range[t0])
                else:
                    t0 = [i-1, i, i+1]
                    temperature_i = []
                    rate_i = []
                    for t_i in t0:
                        temp = self.temperature_range[t_i]
                        self.Temp = self.temperature_range[t_i]
                        # self.calc_k(initial_state=all_result[i][:, -1])
                        rate_all = self.elementeny_equation_all(
                            initial_state=all_result[t_i][:, -1], solve=False)
                        rate0 = rate_all[index_pro]
                        rate_i.append(rate0)
                        temperature_i.append(temp)
                    derivative_backward = (math.log(
                        rate_i[1], math.e)-math.log(rate_i[0], math.e))/(temperature_i[1]-temperature_i[0])
                    derivative_forward = (math.log(
                        rate_i[2], math.e)-math.log(rate_i[1], math.e))/(temperature_i[2]-temperature_i[1])
                    derivative = (derivative_forward+derivative_backward)/2
                apparent_energy[j].append(
                    derivative*self.kB*math.pow(self.temperature_range[i], 2))
        for i in range(len(apparent_energy)):
            apparent_energy_i = apparent_energy[i]
            plt.plot(self.temperature_range, apparent_energy_i, '--')
        nocl = math.ceil(len(self.species) / 6)
        plt.legend(self.product_species, ncol=nocl, frameon=False)
        plt.xlabel('Temperature (K)', fontsize=fontsize_title)
        plt.ylabel('Apparent energy (eV)', fontsize=fontsize_title)
        # plt.show()

        keys = ["Temperature"]
        keys.extend(self.product_species)
        value = copy.deepcopy([self.temperature_range])
        value.extend(apparent_energy)
        dict_apparent_energy = dict(zip(keys, value))
        df = DataFrame(dict_apparent_energy)
        df.to_excel('./%s/apparent_energy.xlsx' % self.filename)
        df.to_csv('./%s/apparent_energy.csv' % self.filename)

        # -----------save------------------
        fig.savefig('./%s/apparent_energy.eps' % self.filename,
                    format='eps',
                    bbox_inches='tight', dpi=300)
        fig.savefig('./%s/apparent_energy.tif' % self.filename,
                    format='tif',
                    bbox_inches='tight', dpi=300)

    def degree_rate_control(self, all_result):
        """
        plot of drc
        @param list[numpy] -> all_result      all coverage data
        """
        print("#---------- start to calculate DRC ----------")
        all_result_ini = all_result
        degree_diff = [0.98, 1.02]
        degree_rate_all = [[] for i in range(len(self.product_species))]

        r_all = [[] for i in range(len(self.product_species))]
        for i in range(len(self.temperature_range)):
            self.Temp = self.temperature_range[i]
            # ------------ k -----------------
            # self.calc_k(initial_state=all_result[i][:, -1])
            rate_all = self.elementeny_equation_all(
                initial_state=all_result_ini[i][:, -1], solve=False)
            for j in range(len(self.index_product_species)):
                index_j = self.index_product_species[j]
                r_all[j].append(rate_all[index_j])

        rate_all_k = [[[r_all[j]]
                       for i in range(len(self.element_step))]
                      for j in range(len(self.index_product_species))]

        for i in range(len(degree_diff)):
            degree_diff_i = degree_diff[i]
            for j in range(len(self.element_step)):
                k_change = np.ones(len(self.element_step))
                k_change[j] *= degree_diff_i
                r_all_e = [[] for i in range(len(self.product_species))]
                all_result_ini = self.odes_solution(k_change=k_change)
                for t in range(len(self.temperature_range)):
                    self.Temp = self.temperature_range[t]
                    # ------------ k -----------------
                    # self.calc_k(k_change=k_change, initial_state=all_result[i][:, -1])
                    rate_all = self.elementeny_equation_all(
                        k_change=k_change, initial_state=all_result_ini[t][:, -1], solve=False)
                    for l in range(len(self.index_product_species)):
                        index_l = self.index_product_species[l]
                        r_all_e[l].append(rate_all[index_l])
                for k in range(len(self.index_product_species)):
                    rate_all_k[k][j].append(r_all_e[k])
        for i in range(len(rate_all_k)):
            for j in range(len(rate_all_k[i])):
                diff0_r = np.log(np.array(rate_all_k[i][j][0]))
                diff1_r = np.log(np.array(rate_all_k[i][j][1]))
                diff2_r = np.log(np.array(rate_all_k[i][j][2]))
                degree_diff_r = ((diff0_r-diff1_r)/(-math.log(degree_diff[0], math.e)) +
                                 (diff2_r-diff0_r)/(math.log(degree_diff[1], math.e)))/2
                degree_rate_all[i].append(degree_diff_r)
        name_all = self.str2reaction_all

        fontsize_title = 10
        fontsize_txt = 8
        fig, ax = plt.subplots(figsize=(4, 3))
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        for i in range(len(degree_rate_all)):
            for j in range(len(degree_rate_all[i])):
                plt.plot(self.temperature_range,
                         degree_rate_all[i][j], '--', label=name_all[j])
        # max_y = [for i in range(len(self.index_product_species))]
        # y_locator = round(max()/5)
        # y_major_locator = plt.MultipleLocator(0.6)
        # ax.yaxis.set_major_locator(y_major_locator)
        nocl = math.ceil(len(name_all) / 4)
        plt.legend(ncol=nocl, frameon=False)
        plt.xlabel('Temperature (K)', fontsize=fontsize_title)
        plt.ylabel('Degree of rate control', fontsize=fontsize_title)
        # plt.show()

        for i in range(len(self.product_species)):
            keys = ["Temperature"]
            keys.extend(self.str2reaction_all)
            value = copy.deepcopy([self.temperature_range])
            value.extend(degree_rate_all[i])
            dict_drc = dict(zip(keys, value))
            df = DataFrame(dict_drc)
            df.to_excel('./%s/drc_%s.xlsx' %
                        (self.filename, self.product_species[i]))
            df.to_csv('./%s/drc_%s.csv' %
                      (self.filename, self.product_species[i]))

        # -----------save------------------
        fig.savefig('./%s/drc.eps' % self.filename,
                    format='eps',
                    bbox_inches='tight', dpi=300)
        fig.savefig('./%s/drc.tif' % self.filename,
                    format='tif',
                    bbox_inches='tight', dpi=300)

    def str2reaction(self):
        """
        write elementary equation 
        return list -> str2reaction_all     all elementary equation 
        """
        str2reaction_all = []
        for i in range(len(self.element_step)):
            element_e = []
            number_e = []
            elementary_e = []
            for j in range(len(self.element_step[i][0])):
                if self.element_step[i][0][j] not in element_e:
                    element_e.append(self.element_step[i][0][j])
                    number_e.append(1)
                else:
                    index_e = element_e.index(self.element_step[i][0][j])
                    number_e[index_e] += 1

            for k in range(len(element_e)):
                if number_e[k] == 1:
                    elementary_e.append(element_e[k])
                else:
                    elementary_e.append(str(number_e[k])+element_e[k])
            reaction_str = (" + ").join(elementary_e)

            element_e = []
            number_e = []
            elementary_e = []
            for j in range(len(self.element_step[i][1])):
                if self.element_step[i][1][j] not in element_e:
                    element_e.append(self.element_step[i][1][j])
                    number_e.append(1)
                else:
                    index_e = element_e.index(self.element_step[i][1][j])
                    number_e[index_e] += 1
            for k in range(len(element_e)):
                # print(number_e[k])
                if number_e[k] == 1:
                    elementary_e.append(element_e[k])
                else:
                    elementary_e.append(str(number_e[k])+element_e[k])
            product_str = (" + ").join(elementary_e)

            str2reaction_all.append(reaction_str+" -> "+product_str)
        return str2reaction_all
