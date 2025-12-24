#include "mppi_cpp.h"
#include <cmath>
#include <algorithm>
#include <numeric>

MPPICpp::MPPICpp(double m_cart, double m_pole, double l, double g, double dt,
                 int K, int T, double lambda, double sigma,
                 const std::vector<double>& Q, double R)
    : m_cart(m_cart), m_pole(m_pole), l(l), g(g), dt(dt),
      K(K), T(T), lambda(lambda), sigma(sigma), Q(Q), R(R),
      rng(std::random_device{}()), normal_dist(0.0, 1.0) {
    reset();
}

void MPPICpp::reset() {
    u = std::vector<double>(T, 0.0);
    costs_history.clear();
}

std::vector<double> MPPICpp::dynamics(const std::vector<double>& state, double F) {
    double x = state[0];
    double theta = state[1];
    double dx = state[2];
    double dtheta = state[3];
    
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double denom = m_cart + m_pole * sin_theta * sin_theta;
    
    double ddx = (F + m_pole * sin_theta * (l * dtheta * dtheta + g * cos_theta)) / denom;
    double ddtheta = (-F * cos_theta - m_pole * l * dtheta * dtheta * cos_theta * sin_theta -
                     (m_cart + m_pole) * g * sin_theta) / (l * denom);
    
    return {dx, dtheta, ddx, ddtheta};
}

double MPPICpp::cost_function(const std::vector<std::vector<double>>& state_traj,
                             const std::vector<double>& control_traj) {
    double cost = 0.0;
    for (int t = 0; t < T; ++t) {
        const auto& state = state_traj[t];
        cost += Q[0] * state[0] * state[0];
        cost += Q[1] * state[1] * state[1];
        cost += Q[2] * state[2] * state[2];
        cost += Q[3] * state[3] * state[3];
        cost += R * control_traj[t] * control_traj[t];
    }
    return cost;
}

double MPPICpp::compute_control(const std::vector<double>& state) {
    // Генерация случайных возмущений
    std::vector<std::vector<double>> epsilon(K, std::vector<double>(T));
    for (int k = 0; k < K; ++k) {
        for (int t = 0; t < T; ++t) {
            epsilon[k][t] = sigma * normal_dist(rng);
        }
    }
    
    std::vector<double> costs(K, 0.0);
    
    // Оценка стоимости для каждой траектории
    for (int k = 0; k < K; ++k) {
        // Пробная траектория управления
        std::vector<double> u_sample(T);
        for (int t = 0; t < T; ++t) {
            u_sample[t] = u[t] + epsilon[k][t];
        }
        
        // Симуляция траектории
        std::vector<std::vector<double>> state_traj(T, std::vector<double>(4));
        std::vector<double> current_state = state;
        
        for (int t = 0; t < T; ++t) {
            state_traj[t] = current_state;
            
            // Интегрирование динамики
            auto deriv = dynamics(current_state, u_sample[t]);
            for (int i = 0; i < 4; ++i) {
                current_state[i] += deriv[i] * dt;
            }
        }
        
        // Вычисление стоимости
        costs[k] = cost_function(state_traj, u_sample);
    }
    
    // Вычисление весов
    double min_cost = *std::min_element(costs.begin(), costs.end());
    std::vector<double> weights(K);
    double sum_weights = 0.0;
    
    for (int k = 0; k < K; ++k) {
        weights[k] = exp(-(costs[k] - min_cost) / lambda);
        sum_weights += weights[k];
    }
    
    // Нормализация весов
    for (int k = 0; k < K; ++k) {
        weights[k] /= sum_weights;
    }
    
    // Обновление оптимальной траектории
    for (int t = 0; t < T; ++t) {
        double update = 0.0;
        for (int k = 0; k < K; ++k) {
            update += weights[k] * epsilon[k][t];
        }
        u[t] += update;
    }
    
    // Сохранение истории
    costs_history.push_back(min_cost);
    
    return u[0];
}