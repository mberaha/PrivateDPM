#include <math.h>

#include <fstream>
#include <iostream>

#include "lib/argparse/argparse.h"
#include "src/includes.h"

std::string CURR_DIR = "privacy_experiments/test_beta_mix/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

Eigen::MatrixXd simulate_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd a_params(3);
  Eigen::VectorXd b_params(3);
  a_params << 5.0, 50.0, 50.0;
  b_params << 50.0, 50.0, 5.0;

  Eigen::MatrixXd out(ndata, 1);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    out(i, 0) =
        stan::math::beta_rng(a_params[c_alloc], b_params[c_alloc], rng);
  }
  return out;
}

Eigen::VectorXd eval_true_dens(Eigen::VectorXd xgrid) {
  Eigen::VectorXd out(xgrid.size());
  for (int i = 0; i < xgrid.size(); i++) {
    double x = xgrid[i];
    if ((x <= 0) || (x >= 1)) {
      out(i) = 0;
    } else {
      out(i) = 1.0 / 3.0 *
               (std::exp(stan::math::beta_lpdf(x, 5.0, 50.0)) +
                std::exp(stan::math::beta_lpdf(x, 50.0, 50.0)) +
                std::exp(stan::math::beta_lpdf(x, 50.0, 5.0)));
    }
  }
  return out;
}


int main() {
  int ndata = 500;

  Eigen::MatrixXd data = simulate_data(ndata);
  auto &factory_algo = AlgorithmFactory::Instance();
  auto &factory_hier = HierarchyFactory::Instance();
  auto &factory_mixing = MixingFactory::Instance();

  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb",
                                 &algo_proto);

  auto algo = factory_algo.create_object(algo_proto.algo_id());
  auto hier = factory_hier.create_object("BetaGG");
  auto mixing =factory_mixing.create_object("DP");
  BaseCollector* coll = new MemoryCollector();

  algo->read_params_from_proto(algo_proto);
  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());
  bayesmix::read_proto_from_file(PARAM_DIR + "bgg_params.asciipb",
                                 hier->get_mutable_prior());
  hier->initialize();

  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_hierarchy(hier);
  
  algo->run(coll);

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.0);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  bayesmix::write_matrix_to_file(dens, OUT_DIR + "eval_dens.csv");
  bayesmix::write_matrix_to_file(grid, OUT_DIR + "dens_grid.csv");

  Eigen::MatrixXi clusterings(coll->get_size(), data.rows());
  for (int i = 0; i < coll->get_size(); i++) {
    bayesmix::AlgorithmState state;
    coll->get_next_state(&state);
    for (int j = 0; j < data.rows(); j++) {
    clusterings(i, j) = state.cluster_allocs(j);
    }
  }
  bayesmix::write_matrix_to_file(clusterings, OUT_DIR + "clus_chain.csv");

  Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clusterings);
  bayesmix::write_matrix_to_file(best_clus, OUT_DIR + "best_clus.csv");

  bayesmix::write_matrix_to_file(data, OUT_DIR + "data.csv");

  delete coll;
}
