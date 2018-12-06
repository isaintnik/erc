#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


const double AVG_TIME_BETWEEN_SESSIONS = 5;


void calc_lambdas(int project_id, py::array_t<double> user_embedding, py::array_t<double> project_embeddings, int dim,
        double beta, py::array_t<double> interactions, bool derivative, double foreign_coefficient,
        py::array_t<int> project_ids, py::array_t<int> n_tasks, py::array_t<double> time_deltas,
        py::array_t<double> out_lambdas, py::array_t<double> out_user_derivatives,
        py::array_t<double> out_project_derivatives) {
    double e = exp(beta);
    double numerator = 10;
    double denominator = 10;
    ssize_t n_sessions = project_ids.size();
    auto project_ids_accessor = project_ids.unchecked<1>();
    auto n_tasks_accessor = n_tasks.unchecked<1>();
    auto interactions_accessor = interactions.unchecked<1>();
    auto out_accessor = out_lambdas.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n_sessions; ++i) {
        const int cur_project_id = project_ids_accessor(i);
        const double coefficient = cur_project_id == project_id ? 1 : foreign_coefficient;
        numerator = e * numerator + coefficient * n_tasks_accessor(i) * interactions_accessor(cur_project_id);
        denominator = e * denominator + coefficient;
        out_accessor(i) = numerator / denominator;
//        if (derivative) {
//
//        }
    }
}

PYBIND11_MODULE(wheel, m) {
    m.def("calc_lambdas", &calc_lambdas);
}
