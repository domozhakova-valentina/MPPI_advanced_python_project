#ifndef MPPI_CPP_H
#define MPPI_CPP_H

#include <vector>
#include <random>
<<<<<<< HEAD

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
=======
#include <memory>
#include <cmath>


struct SystemConfig {
    double cart_mass;      // M - масса тележки
    double pole_mass;      // m - масса маятника
    double pole_length;    // l - длина маятника
    double gravity;        // g - ускорение свободного падения
    double dt;             // шаг времени для дискретизации

    SystemConfig() : cart_mass(1.0), pole_mass(0.1), pole_length(1.0),
                     gravity(9.81), dt(0.02) {}
};


struct MPPIConfig {
    int num_samples;       // K - количество траекторий
    int horizon;           // T - горизонт планирования
    double lambda;         // λ - параметр для вычисления весов
    double noise_sigma;    // σ - стандартное отклонение шума
    double control_limit;  // максимальное значение силы

    MPPIConfig() : num_samples(1000), horizon(30), lambda(0.1),
                   noise_sigma(1.0), control_limit(10.0) {}
};


struct State {
    double x;          // положение тележки
    double theta;      // угол маятника
    double x_dot;      // скорость тележки
    double theta_dot;  // угловая скорость маятника

    State() : x(0.0), theta(0.0), x_dot(0.0), theta_dot(0.0) {}
    State(double x, double theta, double x_dot, double theta_dot)
        : x(x), theta(theta), x_dot(x_dot), theta_dot(theta_dot) {}
};

/**
 * @brief Интерфейс для модели динамики системы
 */
class DynamicsModel {
public:
    virtual ~DynamicsModel() = default;

    /**
     * @brief Вычисляет следующее состояние системы
     * @param state текущее состояние
     * @param control приложенная сила
     * @param dt шаг времени
     * @return следующее состояние
     */
    virtual State step(const State& state, double control, double dt) const = 0;

    /**
     * @brief Вычисляет производные состояния
     * @param state текущее состояние
     * @param control приложенная сила
     * @return производные [dx, dtheta, dx_dot, dtheta_dot]
     */
    virtual std::vector<double> derivatives(
        const State& state, double control) const = 0;
};

/**
 * @brief Модель перевернутого маятника на тележке
 */
class InvertedPendulumModel : public DynamicsModel {
private:
    SystemConfig config_;

public:
    explicit InvertedPendulumModel(const SystemConfig& config);

    State step(const State& state, double control, double dt) const override;
    std::vector<double> derivatives(
        const State& state, double control) const override;
};

/**
 * @brief Основной класс реализации алгоритма MPPI на C++
 */
class MPPIController {
private:
    SystemConfig system_config_;
    MPPIConfig mppi_config_;
    std::unique_ptr<DynamicsModel> model_;

    // Текущая оптимальная траектория управления
    std::vector<double> nominal_controls_;

    // Генератор случайных чисел
    std::mt19937 generator_;
    std::normal_distribution<double> distribution_;

    // Вспомогательные методы
    double computeCost(const std::vector<State>& trajectory,
                      const std::vector<double>& controls) const;
    std::vector<State> rolloutTrajectory(
        const State& initial_state,
        const std::vector<double>& controls) const;

public:
    /**
     * @brief Конструктор контроллера MPPI
     * @param system_config конфигурация системы
     * @param mppi_config конфигурация алгоритма
     */
    MPPIController(const SystemConfig& system_config,
                  const MPPIConfig& mppi_config);

    /**
     * @brief Выполняет один шаг алгоритма MPPI
     * @param current_state текущее состояние системы
     * @return управляющее воздействие (сила)
     */
    double computeControl(const State& current_state);

    /**
     * @brief Сбрасывает контроллер (например, при новой симуляции)
     */
    void reset();

    // Геттеры для конфигурации
    SystemConfig getSystemConfig() const { return system_config_; }
    MPPIConfig getMPPIConfig() const { return mppi_config_; }

    // Сеттеры для конфигурации
    void setSystemConfig(const SystemConfig& config) { system_config_ = config; }
    void setMPPIConfig(const MPPIConfig& config) { mppi_config_ = config; }
};

#endif // MPPI_CPP_H
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
