#include "mppi_cpp.h"
<<<<<<< HEAD
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
=======
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include <random>

using namespace std;

/**
 * @brief Конструктор модели маятника
 */
InvertedPendulumModel::InvertedPendulumModel(const SystemConfig& config)
    : config_(config) {}

/**
 * @brief Вычисляет производные состояния для модели перевернутого маятника
 */
vector<double> InvertedPendulumModel::derivatives(
    const State& state, double control) const {

    // Извлекаем параметры для удобства
    const double M = config_.cart_mass;
    const double m = config_.pole_mass;
    const double l = config_.pole_length;
    const double g = config_.gravity;

    // Извлекаем переменные состояния
    const double theta = state.theta;
    const double theta_dot = state.theta_dot;
    const double F = control;

    // Вычисляем промежуточные величины
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double sin2_theta = sin_theta * sin_theta;

    // Знаменатель, общий для обоих уравнений
    const double denom = M + m * sin2_theta;

    // Вычисляем ускорение тележки (ẍ)
    const double x_ddot = (F + m * sin_theta * (l * theta_dot * theta_dot +
                       g * cos_theta)) / denom;

    // Вычисляем угловое ускорение маятника (θ̈)
    const double theta_ddot = (-F * cos_theta -
                            m * l * theta_dot * theta_dot * cos_theta * sin_theta -
                            (M + m) * g * sin_theta) /
                           (l * denom);

    // Возвращаем производные [ẋ, θ̇, ẍ, θ̈]
    return {state.x_dot, state.theta_dot, x_ddot, theta_ddot};
}

/**
 * @brief Вычисляет следующее состояние системы с помощью метода Рунге-Кутты 4-го порядка
 */
State InvertedPendulumModel::step(
    const State& state, double control, double dt) const {

    // Метод Рунге-Кутты 4-го порядка (RK4)
    auto k1 = derivatives(state, control);

    State state2;
    state2.x = state.x + k1[0] * dt / 2.0;
    state2.theta = state.theta + k1[1] * dt / 2.0;
    state2.x_dot = state.x_dot + k1[2] * dt / 2.0;
    state2.theta_dot = state.theta_dot + k1[3] * dt / 2.0;
    auto k2 = derivatives(state2, control);

    State state3;
    state3.x = state.x + k2[0] * dt / 2.0;
    state3.theta = state.theta + k2[1] * dt / 2.0;
    state3.x_dot = state.x_dot + k2[2] * dt / 2.0;
    state3.theta_dot = state.theta_dot + k2[3] * dt / 2.0;
    auto k3 = derivatives(state3, control);

    State state4;
    state4.x = state.x + k3[0] * dt;
    state4.theta = state.theta + k3[1] * dt;
    state4.x_dot = state.x_dot + k3[2] * dt;
    state4.theta_dot = state.theta_dot + k3[3] * dt;
    auto k4 = derivatives(state4, control);

    // Вычисляем конечное состояние
    State next_state;
    next_state.x = state.x + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * dt / 6.0;
    next_state.theta = state.theta + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * dt / 6.0;
    next_state.x_dot = state.x_dot + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) * dt / 6.0;
    next_state.theta_dot = state.theta_dot + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) * dt / 6.0;

    // Нормализуем угол в диапазон [-π, π]
    while (next_state.theta > M_PI) next_state.theta -= 2.0 * M_PI;
    while (next_state.theta < -M_PI) next_state.theta += 2.0 * M_PI;

    return next_state;
}

/**
 * @brief Конструктор MPPI контроллера
 */
MPPIController::MPPIController(const SystemConfig& system_config,
                             const MPPIConfig& mppi_config)
    : system_config_(system_config),
      mppi_config_(mppi_config),
      generator_(random_device{}()),
      distribution_(0.0, mppi_config.noise_sigma) {

    // Создаем модель динамики
    model_ = make_unique<InvertedPendulumModel>(system_config_);

    // Инициализируем номинальную траекторию управления нулями
    nominal_controls_.resize(mppi_config_.horizon, 0.0);

    cout << "MPPI Controller initialized with:" << endl;
    cout << "  Samples (K): " << mppi_config_.num_samples << endl;
    cout << "  Horizon (T): " << mppi_config_.horizon << endl;
    cout << "  Lambda (λ): " << mppi_config_.lambda << endl;
}

/**
 * @brief Вычисляет стоимость траектории
 */
double MPPIController::computeCost(
    const vector<State>& trajectory,
    const vector<double>& controls) const {

    double total_cost = 0.0;

    for (size_t t = 0; t < trajectory.size(); ++t) {
        const State& state = trajectory[t];

        // Штраф за угол маятника (цель: θ = 0)
        double angle_cost = 10.0 * state.theta * state.theta;

        // Штраф за угловую скорость
        double angular_velocity_cost = 0.1 * state.theta_dot * state.theta_dot;

        // Штраф за положение тележки (цель: x = 0)
        double position_cost = 1.0 * state.x * state.x;

        // Штраф за скорость тележки
        double velocity_cost = 0.1 * state.x_dot * state.x_dot;

        // Штраф за управление (минимизация усилий)
        double control_cost = 0.01 * controls[t] * controls[t];

        // Суммируем стоимость для этого шага
        total_cost += angle_cost + angular_velocity_cost +
                     position_cost + velocity_cost + control_cost;
    }

    return total_cost;
}

/**
 * @brief Прокручивает траекторию на горизонте планирования
 */
vector<State> MPPIController::rolloutTrajectory(
    const State& initial_state,
    const vector<double>& controls) const {

    vector<State> trajectory;
    trajectory.reserve(controls.size());

    State current_state = initial_state;

    for (double control : controls) {
        // Ограничиваем управление
        double clamped_control = max(-mppi_config_.control_limit,
                                    min(mppi_config_.control_limit, control));

        // Вычисляем следующее состояние
        current_state = model_->step(current_state, clamped_control,
                                    system_config_.dt);

        trajectory.push_back(current_state);
    }

    return trajectory;
}

/**
 * @brief Основной метод MPPI - вычисляет управляющее воздействие
 */
double MPPIController::computeControl(const State& current_state) {
    int K = mppi_config_.num_samples;
    int T = mppi_config_.horizon;
    double lambda = mppi_config_.lambda;

    // Векторы для хранения стоимостей и весов
    vector<double> costs(K, 0.0);
    vector<double> weights(K, 0.0);

    // Генерируем случайные возмущения
    vector<vector<double>> noise_samples(K, vector<double>(T));
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < T; ++j) {
            noise_samples[i][j] = distribution_(generator_);
        }
    }

    double min_cost = numeric_limits<double>::max();

    // Параллельный цикл для вычисления стоимостей траекторий
    for (int i = 0; i < K; ++i) {
        // Создаем управляющую последовательность для этой траектории
        vector<double> controls(T);
        for (int j = 0; j < T; ++j) {
            controls[j] = nominal_controls_[j] + noise_samples[i][j];
        }

        // Прокручиваем траекторию
        auto trajectory = rolloutTrajectory(current_state, controls);

        // Вычисляем стоимость
        costs[i] = computeCost(trajectory, controls);

        // Обновляем минимальную стоимость
        if (costs[i] < min_cost) {
            min_cost = costs[i];
        }
    }

    // Вычисляем веса
    double weight_sum = 0.0;
    for (int i = 0; i < K; ++i) {
        // Формула весов: exp(-(cost - min_cost)/λ)
        weights[i] = exp(-(costs[i] - min_cost) / lambda);
        weight_sum += weights[i];
    }

    // Нормализуем веса
    if (weight_sum > 1e-8) {
        for (int i = 0; i < K; ++i) {
            weights[i] /= weight_sum;
        }
    }

    // Обновляем номинальную траекторию управления
    for (int j = 0; j < T; ++j) {
        double weighted_noise_sum = 0.0;

        for (int i = 0; i < K; ++i) {
            weighted_noise_sum += weights[i] * noise_samples[i][j];
        }

        // Обновляем управление: u_new = u + Σ(w_i * ε_i)
        nominal_controls_[j] += weighted_noise_sum;

        // Ограничиваем управление
        nominal_controls_[j] = max(-mppi_config_.control_limit,
                                  min(mppi_config_.control_limit,
                                      nominal_controls_[j]));
    }

    // Сдвигаем траекторию на один шаг (Shift-and-last strategy)
    double control_to_apply = nominal_controls_[0];

    for (int j = 0; j < T - 1; ++j) {
        nominal_controls_[j] = nominal_controls_[j + 1];
    }
    nominal_controls_[T - 1] = 0.0;  // Последний элемент заполняем нулем

    return control_to_apply;
}

/**
 * @brief Сбрасывает контроллер
 */
void MPPIController::reset() {
    fill(nominal_controls_.begin(), nominal_controls_.end(), 0.0);
    generator_.seed(random_device{}());
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
}