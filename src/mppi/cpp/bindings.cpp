#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mppi_cpp.h"

namespace py = pybind11;

PYBIND11_MODULE(mppi_cpp, m) {
    py::class_<MPPICpp>(m, "MPPICpp")
        .def(py::init<double, double, double, double, double,
                      int, int, double, double,
                      std::vector<double>, double>(),
             py::arg("m_cart"), py::arg("m_pole"), py::arg("l"), py::arg("g"), py::arg("dt"),
             py::arg("K"), py::arg("T"), py::arg("lambda"), py::arg("sigma"),
             py::arg("Q"), py::arg("R"))
        .def("reset", &MPPICpp::reset)
        .def("compute_control", &MPPICpp::compute_control)
        .def("get_costs_history", &MPPICpp::get_costs_history);
}