#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "mppi_cpp.h"

namespace py = pybind11;

/**
 * @brief Python-обертка для C++ реализации MPPI
 *
 * Экспортирует классы и функции для использования в Python через Pybind11
 */
PYBIND11_MODULE(mppi_cpp, m) {
    m.doc() = "C++ implementation of MPPI algorithm for inverted pendulum control";

    // Класс SystemConfig
    py::class_<SystemConfig>(m, "SystemConfig")
        .def(py::init<>())
        .def_readwrite("cart_mass", &SystemConfig::cart_mass)
        .def_readwrite("pole_mass", &SystemConfig::pole_mass)
        .def_readwrite("pole_length", &SystemConfig::pole_length)
        .def_readwrite("gravity", &SystemConfig::gravity)
        .def_readwrite("dt", &SystemConfig::dt)
        .def("__repr__", [](const SystemConfig& config) {
            return "SystemConfig(cart_mass=" + std::to_string(config.cart_mass) +
                   ", pole_mass=" + std::to_string(config.pole_mass) +
                   ", pole_length=" + std::to_string(config.pole_length) +
                   ", gravity=" + std::to_string(config.gravity) +
                   ", dt=" + std::to_string(config.dt) + ")";
        });

    // Класс MPPIConfig
    py::class_<MPPIConfig>(m, "MPPIConfig")
        .def(py::init<>())
        .def_readwrite("num_samples", &MPPIConfig::num_samples)
        .def_readwrite("horizon", &MPPIConfig::horizon)
        .def_readwrite("lambda", &MPPIConfig::lambda)
        .def_readwrite("noise_sigma", &MPPIConfig::noise_sigma)
        .def_readwrite("control_limit", &MPPIConfig::control_limit)
        .def("__repr__", [](const MPPIConfig& config) {
            return "MPPIConfig(num_samples=" + std::to_string(config.num_samples) +
                   ", horizon=" + std::to_string(config.horizon) +
                   ", lambda=" + std::to_string(config.lambda) +
                   ", noise_sigma=" + std::to_string(config.noise_sigma) +
                   ", control_limit=" + std::to_string(config.control_limit) + ")";
        });

    // Класс State
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def(py::init<double, double, double, double>())
        .def_readwrite("x", &State::x)
        .def_readwrite("theta", &State::theta)
        .def_readwrite("x_dot", &State::x_dot)
        .def_readwrite("theta_dot", &State::theta_dot)
        .def("__repr__", [](const State& state) {
            return "State(x=" + std::to_string(state.x) +
                   ", theta=" + std::to_string(state.theta) +
                   ", x_dot=" + std::to_string(state.x_dot) +
                   ", theta_dot=" + std::to_string(state.theta_dot) + ")";
        })
        .def("to_list", [](const State& state) {
            return py::make_tuple(state.x, state.theta, state.x_dot, state.theta_dot);
        })
        .def_static("from_list", [](py::list lst) {
            return State(
                lst[0].cast<double>(),
                lst[1].cast<double>(),
                lst[2].cast<double>(),
                lst[3].cast<double>()
            );
        });

    // Класс InvertedPendulumModel
    py::class_<InvertedPendulumModel, DynamicsModel>(m, "InvertedPendulumModel")
        .def(py::init<const SystemConfig&>())
        .def("step", &InvertedPendulumModel::step)
        .def("derivatives", &InvertedPendulumModel::derivatives);

    // Основной класс MPPIController
    py::class_<MPPIController>(m, "MPPIController")
        .def(py::init<const SystemConfig&, const MPPIConfig&>())
        .def("compute_control", &MPPIController::computeControl,
             "Compute control action for current state")
        .def("reset", &MPPIController::reset,
             "Reset controller to initial state")
        .def("get_system_config", &MPPIController::getSystemConfig,
             "Get current system configuration")
        .def("get_mppi_config", &MPPIController::getMPPIConfig,
             "Get current MPPI configuration")
        .def("set_system_config", &MPPIController::setSystemConfig,
             "Set system configuration")
        .def("set_mppi_config", &MPPIController::setMPPIConfig,
             "Set MPPI configuration")
        .def("__repr__", [](const MPPIController& controller) {
            auto sys_config = controller.getSystemConfig();
            auto mppi_config = controller.getMPPIConfig();
            return "MPPIController(cart_mass=" + std::to_string(sys_config.cart_mass) +
                   ", K=" + std::to_string(mppi_config.num_samples) +
                   ", T=" + std::to_string(mppi_config.horizon) + ")";
        });

    // Вспомогательные функции
    m.def("create_default_controller", []() {
        SystemConfig sys_config;
        MPPIConfig mppi_config;
        return MPPIController(sys_config, mppi_config);
    }, "Create controller with default parameters");

    m.def("simulate_step", [](MPPIController& controller, const State& state) {
        double control = controller.computeControl(state);
        return py::make_tuple(control, controller.getMPPIConfig());
    }, "Perform one MPPI step and return control action");
}