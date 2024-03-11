#include <chrono>

#include "../utils.h"
#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/algorithms/private_neal2.h"
#include "src/privacy/channels/gaussian_channel.h"
#include "src/privacy/channels/laplace_channel.h"
#include "src/privacy/hierarchies/truncated_nnig_hier.h"


std::string CURR_DIR = "privacy_experiments/blood_donors/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";



void save_stuff_to_file(int ndata, double eps_sq, BaseCollector* coll,
                        std::shared_ptr<BaseAlgorithm> algo,
                        std::string algo_name, double algo_is_private, std::string mechanism) {
  std::string base_fname =
      OUT_DIR + algo_name + mechanism + "_";

  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 10.0);
  Eigen::MatrixXd dens;
  std::cout << "here1" << std::endl;
  if (algo_is_private) {
    dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  } else {
    dens = eval_private_nnig_lpdf(algo, coll, grid, eps_sq, 1);
  }
  std::cout << "here2" << std::endl;

  bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");

  Eigen::MatrixXi clus_allocs = get_cluster_mat(coll, ndata);
  bayesmix::write_matrix_to_file(clus_allocs, base_fname + "clus_chain.csv");

  Eigen::VectorXi best_clus = bayesmix::cluster_estimate(clus_allocs);
  bayesmix::write_matrix_to_file(best_clus, base_fname + "best_clus.csv");
}

std::shared_ptr<PrivateNeal2> get_private_neal2() {
  auto& factory_hier = HierarchyFactory::Instance();
  auto hier = factory_hier.create_object("NNIG");
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");
  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());

  auto algo = std::make_shared<PrivateNeal2>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

std::shared_ptr<PrivateConditionalAlgorithm<SliceSampler>> get_private_slice() {
  auto& factory_hier = HierarchyFactory::Instance();
  auto hier = factory_hier.create_object("NNIG");
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("TruncSB");
  bayesmix::read_proto_from_file(PARAM_DIR + "truncsb.asciipb",
                                 mixing->get_mutable_prior());

  auto algo = std::make_shared<PrivateConditionalAlgorithm<SliceSampler>>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
//   algo->set_verbose(false);
  return algo;
}

std::shared_ptr<Neal2Algorithm> get_neal2(double eps_sq) {
  auto hier = std::make_shared<PrivateNIGHier>();
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->set_var_bounds(eps_sq);
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");

  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());
  auto algo = std::make_shared<Neal2Algorithm>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

std::shared_ptr<Neal2Algorithm> get_neal3(double eps_sq) {
  auto hier = std::make_shared<PrivateNIGHier>();
  bayesmix::read_proto_from_file(PARAM_DIR + "nnig_ngg.asciipb",
                                 hier->get_mutable_prior());
  hier->set_var_bounds(eps_sq);
  hier->initialize();

  auto& factory_mixing = MixingFactory::Instance();
  auto mixing = factory_mixing.create_object("DP");

  bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                 mixing->get_mutable_prior());
  auto algo = std::make_shared<Neal3Algorithm>();
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);
  algo->read_params_from_proto(algo_proto);

  algo->read_params_from_proto(algo_proto);
  algo->set_hierarchy(hier);
  algo->set_mixing(mixing);
  algo->set_verbose(false);
  return algo;
}

int main() {

    Eigen::MatrixXd sanitized_data_gauss =
      bayesmix::read_eigen_matrix("privacy_experiments/data/blood_donors_gauss_priv.csv");
    int ndata = sanitized_data_gauss.rows();
    double eps_gauss = std::sqrt(0.0036);


    Eigen::MatrixXd sanitized_data_lap =
      bayesmix::read_eigen_matrix("privacy_experiments/data/blood_donors_laplace_priv.csv");
    std::cout << sanitized_data_lap.transpose() << std::endl;
    // int ndata = sanitized_data_lap.rows();
    double eps_lap = std::sqrt(0.03);

    auto& factory_hier = HierarchyFactory::Instance();
    auto& factory_mixing = MixingFactory::Instance();

    // auto neal2 = get_neal2(eps_gauss * eps_gauss);
    // neal2->set_data(sanitized_data_gauss);
    // BaseCollector* neal2coll = new MemoryCollector();
    // neal2->run(neal2coll);
    // save_stuff_to_file(ndata, eps_gauss * eps_gauss, neal2coll,
    //                     neal2, "neal2", false, "gaussian");
    // delete neal2coll;

    // auto neal3 = get_neal3(eps_gauss * eps_gauss);
    // neal3->set_data(sanitized_data_gauss);
    // BaseCollector* neal3coll = new MemoryCollector();
    // neal3->run(neal3coll);
    // save_stuff_to_file(ndata, eps_gauss * eps_gauss, neal3coll,
    //                     neal3, "neal3", false, "gaussian");
    // delete neal3coll;

    std::shared_ptr<GaussianChannel> channel(new GaussianChannel(eps_gauss));
    std::shared_ptr<LaplaceChannel> lap_channel(new LaplaceChannel(eps_lap));

    auto privateslice = get_private_slice();
    privateslice->set_channel(channel);
    privateslice->set_public_data(sanitized_data_gauss);
    BaseCollector* slicecoll = new MemoryCollector();
    privateslice->run(slicecoll);
    save_stuff_to_file(ndata, eps_gauss * eps_gauss, slicecoll,
                       privateslice, "slice", true, "gaussian");
    delete slicecoll;

    auto privateslice_lap = get_private_slice();
    privateslice_lap->set_channel(lap_channel);
    privateslice_lap->set_public_data(sanitized_data_lap);
    BaseCollector* privateslicecoll_lap = new MemoryCollector();
    privateslice_lap->run(privateslicecoll_lap);
    save_stuff_to_file(ndata, eps_lap,
                     privateslicecoll_lap, privateslice_lap, "slice", true, "laplace");
    delete privateslicecoll_lap;


    // auto privateneal2 = get_private_neal2();
    // privateneal2->set_channel(channel);
    // privateneal2->set_public_data(sanitized_data_gauss);
    // BaseCollector* privneal2coll = new MemoryCollector();
    // privateneal2->run(privneal2coll);
    // save_stuff_to_file(ndata, eps_gauss * eps_gauss,
    //                     privneal2coll, privateneal2, "privateneal2", true, "gaussian");
    // delete privneal2coll;

    // auto privateneal2_lap = get_private_neal2();
    // privateneal2_lap->set_channel(lap_channel);
    // privateneal2_lap->set_public_data(sanitized_data_lap);
    // BaseCollector* privneal2coll_lap = new MemoryCollector();
    // privateneal2_lap->run(privneal2coll_lap);
    // save_stuff_to_file(ndata, eps_lap,
    //                  privneal2coll_lap, privateneal2, "privateneal2", true, "laplace");
    // delete privneal2coll_lap;


}
