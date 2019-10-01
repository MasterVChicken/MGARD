#include "catch2/catch.hpp"

#include "moab/Core.hpp"

#include <cmath>

#include "blaspp/blas.hh"

#include "MassMatrix.hpp"
#include "pcg.hpp"

#include "testing_utilities.hpp"

TEST_CASE("mass matrix and mass matrix preconditioner", "[MassMatrix]") {
    SECTION("triangles") {
        moab::ErrorCode ecode;
        const std::size_t num_nodes = 5;
        const std::size_t num_edges = 7;
        const std::size_t num_tris = 3;
        moab::Core mbcore;
        ecode = mbcore.load_file(mesh_path("lopsided.msh").c_str());
        require_moab_success(ecode);
        mgard::MeshLevel mesh(mbcore);
        REQUIRE(mesh.topological_dimension == 2);
        REQUIRE(mesh.element_type == moab::MBTRI);
        REQUIRE(mesh.ndof() == num_nodes);
        REQUIRE(mesh.entities[moab::MBEDGE].size() == num_edges);
        REQUIRE(mesh.entities[moab::MBTRI].size() == num_tris);
        mgard::MassMatrix M(&mesh);
        mgard::MassMatrixPreconditioner P(&mesh);

        SECTION("preconditioner") {
            double v[num_nodes] = {3, -1, 2, -4, -5};
            double b[num_nodes];
            double expected[num_nodes] = {
                v[0] / 10.5,
                v[1] / 22.5,
                v[2] / 18,
                v[3] / 12,
                v[4] / 4.5
            };
            P(v, b);
            bool all_close = true;
            for (std::size_t i = 0; i < num_nodes; ++i) {
                all_close = all_close && b[i] == Approx(expected[i]);
            }
            REQUIRE(all_close);
        }

        SECTION("mass matrix") {
            //These products were obtained using the Python implementation.
            const std::size_t num_trials = 8;
            const double vs[num_trials][num_nodes] = {
                {1, 0, 0, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 0, 1, 0, 0},
                {0, 0, 0, 1, 0},
                {0, 0, 0, 0, 1},
                {1, -1, 6, 0, -1},
                {15.1, 6.7, 0.4, -8.3, 30.4},
                {108.5, 90.4, -32.7, -61.1, -12.1}
            };
            const double expecteds[num_trials][num_nodes] = {
                {1.75, 0.875, 0.5, 0., 0.375},
                {0.875, 3.75, 1.5, 1., 0.375},
                {0.5, 1.5, 3., 1., 0.},
                {0., 1., 1., 2., 0.},
                {0.375, 0.375, 0., 0., 0.75},
                {3.5, 5.75, 17., 5., -0.75},
                {43.8875, 42.0375, 10.5, -9.5, 30.975},
                {248.0875, 319.25, 30.65, -64.5, 65.5125}
            };

            for (std::size_t i = 0; i < num_trials; ++i) {
                double b[num_nodes];
                M(vs[i], b);
                bool all_close = true;
                for (std::size_t j = 0; j < num_nodes; ++j) {
                    all_close = all_close && b[j] == Approx(expecteds[i][j]);
                }
                REQUIRE(all_close);

                double w[num_nodes];
                double buffer[4 * num_nodes];
                helpers::pcg(M, b, P, w, buffer);
                for (std::size_t j = 0; j < num_nodes; ++j) {
                    all_close = all_close && w[j] == Approx(vs[i][j]);
                }
            }
        }
    }

    SECTION("tetrahedra") {
        moab::ErrorCode ecode;
        moab::Core mbcore;
        const std::size_t num_nodes = 5;
        const std::size_t num_edges = 9;
        const std::size_t num_tets = 2;
        ecode = mbcore.load_file(mesh_path("hexahedron.msh").c_str());
        require_moab_success(ecode);
        mgard::MeshLevel mesh(mbcore);
        REQUIRE(mesh.topological_dimension == 3);
        REQUIRE(mesh.element_type == moab::MBTET);
        REQUIRE(mesh.ndof() == num_nodes);
        REQUIRE(mesh.entities[moab::MBEDGE].size() == num_edges);
        REQUIRE(mesh.entities[moab::MBTET].size() == num_tets);
        double u[num_nodes] = {-8, -7, 3, 0, 10};
        double b[num_nodes];
        mgard::MassMatrix M(&mesh);
        M(u, b);
        mgard::MassMatrixPreconditioner P(&mesh);
        const moab::Range &elements = mesh.entities[mesh.element_type];
        double measures[num_tets];
        for (std::size_t i = 0; i < num_tets; ++i) {
            measures[i] = mesh.measure(elements[i]);
        }

        SECTION("mass matrix") {
            {
                double expected[num_nodes] = {
                    -20 * measures[0],
                    -19 * measures[0] - 1 * measures[1],
                    -9 * measures[0] + 9 * measures[1],
                    -12 * measures[0] + 6 * measures[1],
                    16 * measures[1]
                };
                blas::scal(num_nodes, 1.0 / 20.0, expected, 1);
                bool all_close = true;
                for (std::size_t i = 0; i < num_nodes; ++i) {
                    all_close = all_close && b[i] == Approx(expected[i]);
                }
                REQUIRE(all_close);
            }
        }

        SECTION("preconditioner") {
            double expected[num_nodes];
            for (std::size_t i = 0; i < num_nodes; ++i) {
                expected[i] = b[i];
            }
            expected[0] /= measures[0];
            for (std::size_t i = 1; i < 4; ++i) {
                expected[i] /= (measures[0] + measures[1]);
            }
            expected[4] /= measures[1];
            double v[num_nodes];
            P(b, v);
            bool all_close = true;
            for (std::size_t i = 0; i < num_nodes; ++i) {
                all_close = all_close && v[i] == Approx(expected[i]);
            }
            REQUIRE(all_close);
        }

        {
            double w[num_nodes];
            double buffer[4 * num_nodes];
            helpers::pcg(M, b, P, w, buffer);

            double result[num_nodes];
            M(w, result);

            blas::copy(num_nodes, u, 1, buffer, 1);
            blas::axpy(num_nodes, -1, w, 1, buffer, 1);
            double square_relative_error = (
                blas::nrm2(num_nodes, buffer, 1) / blas::nrm2(num_nodes, u, 1)
            );
            REQUIRE(std::abs(square_relative_error) < 1e-6);
        }
    }
}