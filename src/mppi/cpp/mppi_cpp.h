#ifndef MPPI_CPP_H
#define MPPI_CPP_H

#include <vector>
#include <random>

class MPPICpp {
private:
    // Параметры
    double m_cart;
    double m_pole;
    double l;
    double g;
    double dt;
    
    // Параметры MPPI
    int K;
    int T;
    double lambda;
    double sigma;
    std::vector<double> Q;
    double R;
    
    // Состояние
    std::vector<double> u;
    std::vector<double> costs_history;
    
    // Генератор случайных чисел
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    
    // Вспомогательные методы
    std::vector<double> dynamics(const std::vector<double>& state, double F);
    double cost_function(const std::vector<std::vector<double>>& state_traj,
                        const std::vector<double>& control_traj);
    
public:
    MPPICpp(double m_cart, double m_pole, double l, double g, double dt,
            int K, int T, double lambda, double sigma,
            const std::vector<double>& Q, double R);
    
    void reset();
    double compute_control(const std::vector<double>& state);
    
    const std::vector<double>& get_costs_history() const { return costs_history; }
};

#endif