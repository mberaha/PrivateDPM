#include <chrono>

#include "../utils.h"
#include "src/includes.h"
#include "src/privacy/algorithms/private_conditional.h"
#include "src/privacy/channels/laplace_channel.h"
#include "src/privacy/channels/smoothed_histogram_sampler_channel.h"


std::string CURR_DIR = "privacy_experiments/global/";
std::string OUT_DIR = CURR_DIR + "out/";
std::string PARAM_DIR = CURR_DIR + "params/";

std::pair<Eigen::MatrixXd, Eigen::VectorXi> simulate_private_data(int ndata) {
  Eigen::VectorXd probs = Eigen::VectorXd::Ones(3) / 3.0;
  Eigen::VectorXd a_params(3);
  Eigen::VectorXd b_params(3);
  a_params << 5.0, 50.0, 50.0;
  b_params << 50.0, 50.0, 5.0;

  Eigen::MatrixXd out(ndata, 1);
  Eigen::VectorXi clus(ndata);
  auto& rng = bayesmix::Rng::Instance().get();
  for (int i = 0; i < ndata; i++) {
    int c_alloc = bayesmix::categorical_rng(probs, rng, 0);
    clus(i) = c_alloc;
    out(i, 0) =
        stan::math::beta_rng(a_params[c_alloc], b_params[c_alloc], rng);
  }
  return std::make_pair(out, clus);
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

void save_stuff_to_file(int ndata, double alpha, int repnum,
                        Eigen::MatrixXd private_data,
                        BaseCollector* coll,
                        std::shared_ptr<BaseAlgorithm> algo,
                        std::string channel_name, std::string algo_name) {
  std::string base_fname;
  base_fname = OUT_DIR + "_" + algo_name + "_" + channel_name + "_ndata_" 
              + std::to_string(ndata) + "_alpha_" + std::to_string(alpha) + +"_rep_" +
                std::to_string(repnum) + "_"; 
  
  std::cout << "evaluating the density" << std::endl;
  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.0);
  Eigen::MatrixXd dens = bayesmix::eval_lpdf_parallel(algo, coll, grid);
  std::cout << "done" << std::endl;

  if (repnum == 0) {
    bayesmix::write_matrix_to_file(dens, base_fname + "eval_dens.csv");
  }
  std::cout << "written to file " << std::endl;

  Eigen::MatrixXd clus_allocs = get_cluster_mat(coll, ndata);

  Eigen::VectorXi nclus_chain(clus_allocs.rows());
  for (int j = 0; j < clus_allocs.rows(); j++) {
    Eigen::VectorXi curr_clus = clus_allocs.row(j);
    std::set<int> uniqs{curr_clus.data(), curr_clus.data() + curr_clus.size()};
    nclus_chain(j) = uniqs.size();
  }
  bayesmix::write_matrix_to_file(nclus_chain, base_fname + "nclus_chain.csv");

  Eigen::MatrixXd arate(1, 1);
  if (algo_name == "neal2") {
    arate(0, 0) =
        std::dynamic_pointer_cast<PrivateNeal2>(algo)->get_acceptance_rate();
  } else if (algo_name == "slice") {
    arate(0, 0) =
        std::dynamic_pointer_cast<PrivateConditionalAlgorithm<SliceSampler>>(
            algo)
            ->get_acceptance_rate();
  } else {
    throw std::runtime_error("Unknown algorithm name");
  }
  bayesmix::write_matrix_to_file(arate, base_fname + "acceptance_rate.csv");
}

void run_experiment(int ndata, int repnum) {
  auto [private_data, clus] = simulate_private_data(ndata);
  bayesmix::write_matrix_to_file(
      clus, OUT_DIR + "ndata_" + std::to_string(ndata) + "_rep_" +
                std::to_string(repnum) + "trueclus.csv");

  auto& factory_hier = HierarchyFactory::Instance();
  auto& factory_mixing = MixingFactory::Instance();
  std::vector<double> priv_levels = {2.0, 10.0, 50.0, 100.0, 250.0};
  bayesmix::AlgorithmParams algo_proto;
  bayesmix::read_proto_from_file(PARAM_DIR + "algo.asciipb", &algo_proto);

  for (int k = 0; k < priv_levels.size(); k++) {
    double alpha = priv_levels[k];

    // HISTOGRAM CHANNEL
    double n = private_data.rows();
    int L = 2;
    int nbins = L * std::floor(std::pow(n, 1.0 / 5.0) + 1);
    int npub = std::floor(std::pow(n, 3.0 / 5.0) + 1);

    // compute optimal delta as the minimum satisfying eq (6) in Wasserman and Zhou
    auto delta_grid = Eigen::VectorXd::LinSpaced(200, 1e-10, 0.5);
    auto tmp = (Eigen::VectorXd::Ones(200) - delta_grid).array() / delta_grid.array() * (nbins / n) + 1.0;
    auto cond = tmp.log() < alpha / npub;
    std::cout << cond.transpose() << std::endl;
    double delta = 1.0;
    for (int pos = 0; pos < delta_grid.size(); pos ++) {
        if (cond[pos] == 1) {
            delta = delta_grid[pos];
            break;
        }
    }
    std::shared_ptr<SmoothedHistogramSamplerChannel> hist_channel(
        new SmoothedHistogramSamplerChannel(0, 1, nbins, npub, delta));
    Eigen::MatrixXd hist_sanitized_data = hist_channel->sanitize(private_data);
    
    auto cond_algo =
        std::make_shared<PrivateConditionalAlgorithm<SliceSampler>>();
    cond_algo->read_params_from_proto(algo_proto);
    std::cout << "algo" << std::endl << algo_proto.DebugString() << std::endl;
    auto cond_mixing = factory_mixing.create_object("TruncSB");
    auto cond_hier = factory_hier.create_object("BetaGG");
    bayesmix::read_proto_from_file(PARAM_DIR + "truncsb.asciipb",
                                    cond_mixing->get_mutable_prior());
    bayesmix::read_proto_from_file(PARAM_DIR + "bgg_params.asciipb",
                                     cond_hier->get_mutable_prior());
    cond_hier->initialize();
    std::cout << cond_hier->get_mutable_prior()->DebugString() << std::endl;
    cond_algo->set_hierarchy(cond_hier);
    cond_algo->set_mixing(cond_mixing);
    // cond_algo->set_verbose(false);
    cond_algo->set_channel(hist_channel);
    cond_algo->set_public_data(hist_sanitized_data, ndata, true);

    BaseCollector* cond_coll = new MemoryCollector();
    cond_algo->run(cond_coll);
    save_stuff_to_file(ndata, alpha, repnum, private_data, cond_coll,
                       cond_algo, "histogram", "slice");

    delete cond_coll;



    auto marg_algo =
        std::make_shared<PrivateNeal2>();
    marg_algo->read_params_from_proto(algo_proto);
    std::cout << "algo" << std::endl << algo_proto.DebugString() << std::endl;
    auto marg_mixing = factory_mixing.create_object("DP");
    auto marg_hier = factory_hier.create_object("BetaGG");
    bayesmix::read_proto_from_file(PARAM_DIR + "dp_gamma.asciipb",
                                    marg_mixing->get_mutable_prior());
    bayesmix::read_proto_from_file(PARAM_DIR + "bgg_params.asciipb",
                                     marg_hier->get_mutable_prior());
    marg_hier->initialize();
    std::cout << marg_hier->get_mutable_prior()->DebugString() << std::endl;
    marg_algo->set_hierarchy(marg_hier);
    marg_algo->set_mixing(marg_mixing);
    // marg_algo->set_verbose(false);
    marg_algo->set_channel(hist_channel);
    marg_algo->set_public_data(hist_sanitized_data, ndata, true);

    BaseCollector* marg_coll = new MemoryCollector();
    marg_algo->run(marg_coll);
    save_stuff_to_file(ndata, alpha, repnum, private_data, marg_coll,
                       marg_algo, "histogram", "neal2");

    delete marg_coll;
  }
}


int main() {
  int ndata = 250;
  int nrep = 1;

// #pragma omp parallel for
  for (int i = 0; i < nrep; i++) {
    // {
// #pragma omp critical
//       std::cout << "repnum: " << i << std::endl;
//     }
      run_experiment(ndata, i);
  }
}
