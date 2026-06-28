// cpp/option_pricer.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>


// External references for the benchmark prices

// S0=100, K=100, r=0.05, sigma=0.2, T=1
constexpr double EUROPEAN_REFERENCE = 10.450583572185565; // Black-Scholes price for European option

// S0=100, K=100, r=0.05, sigma=0.2, T=1, n_steps=252
constexpr double ASIAN_REFERENCE = 5.781813651077403; // Reference price for Asian option


// Offsets for European and Asian options
constexpr int EUROPEAN_OFFSET = 100000;
constexpr int ASIAN_OFFSET = 200000;

// Option constants
constexpr double S0 = 100.0;
constexpr double K = 100.0;
constexpr double R = 0.05;
constexpr double SIGMA = 0.20;
constexpr double T = 1.0;
constexpr int N_STEPS = 252;
constexpr int N_REPLICATIONS = 20;
constexpr unsigned int SEED = 0;

// Path Grid
std::vector<int> PATH_GRID{
    static_cast<int>(std::pow(2, 8)),
    static_cast<int>(std::pow(2, 9)),
    static_cast<int>(std::pow(2, 10)),
    static_cast<int>(std::pow(2, 11)),
    static_cast<int>(std::pow(2, 12)),
    static_cast<int>(std::pow(2, 13)),
    static_cast<int>(std::pow(2, 14))
};


struct OptionParams
{
    double S0;
    double K;
    double r;
    double sigma;
    double T;
};


struct EstimatorStats
{
    double price;
    double std_error;
    double ci_low;
    double ci_high;
};


/*
Compute the arithmetic mean of a vector of values.

Inputs:
    values: vector of doubles

Returns:
    The sample mean.
*/
double mean(const std::vector<double>& values)
{
    if(values.empty()){
        throw std::invalid_argument("Vector cannot be empty");
    }
    double sum = 0.0;
    for(const auto& value : values){
        sum += value;
    }
    return sum / values.size();
}


/*
Compute the unbiased sample variance of a vector of values.

Inputs:
    values: vector of doubles

Returns:
    The sample variance.
*/
double sample_variance(const std::vector<double>& values)
{
    if(values.size() < 2){
        throw std::invalid_argument("Need at least two observations");
    }
    double mu = mean(values);
    double sum = 0.0;
    for(const auto& value : values){
        sum += (value-mu) * (value-mu);
    }
    return sum / (values.size() - 1);
}


/*
Compute the median of a vector of values.

Inputs:
    values: vector of doubles

Returns:
    The median.
*/
double median(std::vector<double> values)
{
    if(values.empty()){
        throw std::invalid_argument("Vector cannot be empty");
    }

    std::sort(values.begin(), values.end());

    size_t n = values.size();

    if(n % 2 == 1){
        return values[n / 2];
    }

    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}


/*
Compute Monte Carlo summary statistics from discounted payoffs.

Inputs:
    discounted_payoffs: vector of discounted payoff values

Returns:
    EstimatorStats containing:
        price
        std_error
        ci_low
        ci_high
*/
EstimatorStats compute_estimator_statistics(const std::vector<double>& discounted_payoffs)
{
    double price = mean(discounted_payoffs);
    double std_error;
    if(discounted_payoffs.size() > 1){
        std_error = std::sqrt(sample_variance(discounted_payoffs)) / std::sqrt(discounted_payoffs.size());
    }
    else{
        std_error = 0.0;
    }
    double ci_low = price - 1.96 * std_error;
    double ci_high = price + 1.96 * std_error;
    return {price, std_error, ci_low, ci_high};
}


/*
Compute the European call payoff.

Inputs:
    ST: terminal stock price
    K: strike price

Returns:
    European call payoff
*/
double european_call_payoff(double ST, double K)
{
    return std::max(ST - K, 0.0);
}


/*
Compute the arithmetic-average Asian call payoff max(average - K, 0).

Inputs:
    path: vector of stock prices along one simulated path
    K: strike price

Returns:
    Asian arithmetic call payoff
*/
double asian_arithmetic_call_payoff(const std::vector<double>& path, double K)
{
    double average = mean(path);
    return std::max(average - K, 0.0);
}


/*
Simulate one terminal GBM stock price under the risk-neutral measure.

Inputs:
    params: option parameters
    rng: Mersenne Twister random number generator

Returns:
    Simulated terminal stock price ST
*/
double simulate_gbm_terminal(const OptionParams& params, std::mt19937& rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    double Z = normal(rng);
    
    double S0 = params.S0;
    double r = params.r;
    double sigma = params.sigma;
    double T = params.T;

    return S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
}


/*
Simulate one full GBM path and return the path values.

Inputs:
    params: option parameters
    n_steps: number of time steps
    rng: Mersenne Twister random number generator

Returns:
    Vector of stock prices including S0 and all simulated time points
*/
std::vector<double> simulate_gbm_path(const OptionParams& params, int n_steps, std::mt19937& rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    double S0 = params.S0;
    double r = params.r;
    double sigma = params.sigma;
    double T = params.T;

    double dt = T / static_cast<double>(n_steps);

    std::vector<double> increments(n_steps);
    for(int i = 0; i < n_steps; i++){
        double Z = normal(rng);
        increments[i] = (r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z;
    }

    std::vector<double> log_paths(n_steps);
    std::partial_sum(increments.begin(), increments.end(), log_paths.begin());

    std::vector<double> final_paths(n_steps + 1);
    final_paths[0] = S0;
    for(int i = 1; i <= n_steps; i++){
        final_paths[i] = S0 * std::exp(log_paths[i-1]);
    }
    
    return final_paths;
}


/*
Price a European call option using standard Monte Carlo.

Inputs:
    params: option parameters
    n_paths: number of Monte Carlo paths
    seed: random seed

Returns:
    EstimatorStats for the discounted European call price
*/
EstimatorStats mc_price_european_call(const OptionParams& params, int n_paths, unsigned int seed)
{
    std::mt19937 rng(seed);
    
    std::vector<double> payoffs(n_paths);
    for(int i = 0; i < n_paths; i++){
        payoffs[i] = simulate_gbm_terminal(params, rng);
    }

    std::vector<double> discounted_payoffs(n_paths);
    for(int i = 0; i < n_paths; i++){
        discounted_payoffs[i] = std::exp(-params.r * params.T) * european_call_payoff(payoffs[i], params.K);
    }
    
    return compute_estimator_statistics(discounted_payoffs);
}


/*
Price an arithmetic-average Asian call option using standard Monte Carlo.

Inputs:
    params: option parameters
    n_paths: number of Monte Carlo paths
    n_steps: number of time steps per path
    seed: random seed

Returns:
    EstimatorStats for the discounted Asian call price
*/
EstimatorStats mc_price_asian_call(const OptionParams& params, int n_paths, int n_steps, unsigned int seed)
{
    std::mt19937 rng(seed);

    std::vector<double> payoffs(n_paths);
    for(int i = 0; i < n_paths; i++){
        std::vector<double> path = simulate_gbm_path(params, n_steps, rng);
        payoffs[i] = asian_arithmetic_call_payoff(path, params.K);
    }

    std::vector<double> discounted_payoffs(n_paths);
    for(int i = 0; i < n_paths; i++){
        discounted_payoffs[i] = std::exp(-params.r * params.T) * payoffs[i];
    }
    
    return compute_estimator_statistics(discounted_payoffs);
}


/*
Write the CSV header for the European benchmark file.

Inputs:
    out: output file stream

Returns:
    None
*/
void write_european_csv_header(std::ofstream& out)
{
    out
        << "option_type,"
        << "n_paths,"
        << "reference_price,"
        << "mean_price,"
        << "std_price,"
        << "mean_abs_error,"
        << "mean_signed_error,"
        << "mean_std_error,"
        << "mean_ci_width,"
        << "mean_runtime_sec,"
        << "median_runtime_sec\n";
}


/*
Write the CSV header for the Asian benchmark file.

Inputs:
    out: output file stream

Returns:
    None
*/
void write_asian_csv_header(std::ofstream& out)
{
    out
        << "option_type,"
        << "n_steps,"
        << "n_paths,"
        << "reference_price,"
        << "mean_price,"
        << "std_price,"
        << "mean_abs_error,"
        << "mean_signed_error,"
        << "mean_std_error,"
        << "mean_ci_width,"
        << "mean_runtime_sec,"
        << "median_runtime_sec\n";
}


/*
Write one European benchmark row to CSV.

Inputs:
    out: output file stream
    n_paths: number of paths used
    replication: replication index
    stats: Monte Carlo output statistics
    runtime_sec: runtime in seconds

Returns:
    None
*/
void write_european_csv_row(
    std::ofstream& out,
    int n_paths,
    double reference_price,
    double mean_price,
    double std_price,
    double mean_abs_error,
    double mean_signed_error,
    double mean_std_error,
    double mean_ci_width,
    double mean_runtime,
    double median_runtime
)
{
    out
        << "european_cpp,"
        << n_paths << ","
        << reference_price << ","
        << mean_price << ","
        << std_price << ","
        << mean_abs_error << ","
        << mean_signed_error << ","
        << mean_std_error << ","
        << mean_ci_width << ","
        << mean_runtime << ","
        << median_runtime
        << "\n";
}


/*
Write one Asian benchmark row to CSV.

Inputs:
    out: output file stream
    n_paths: number of paths used
    n_steps: number of steps per path
    replication: replication index
    stats: Monte Carlo output statistics
    runtime_sec: runtime in seconds

Returns:
    None
*/
void write_asian_csv_row(
    std::ofstream& out,
    int n_steps,
    int n_paths,
    double reference_price,
    double mean_price,
    double std_price,
    double mean_abs_error,
    double mean_signed_error,
    double mean_std_error,
    double mean_ci_width,
    double mean_runtime,
    double median_runtime
)
{
    out
        << "asian_cpp,"
        << n_steps << ","
        << n_paths << ","
        << reference_price << ","
        << mean_price << ","
        << std_price << ","
        << mean_abs_error << ","
        << mean_signed_error << ","
        << mean_std_error << ","
        << mean_ci_width << ","
        << mean_runtime << ","
        << median_runtime
        << "\n";
}


/*
Run a European benchmark sweep and save all results to CSV.

Inputs:
    params: option parameters
    path_grid: list of path counts to test
    n_replications: number of repetitions per path count
    output_file: CSV output path
    seed: base random seed

Returns:
    None
*/
void run_european_benchmark(
    const OptionParams& params,
    const std::vector<int>& path_grid,
    int n_replications,
    const std::string& output_file,
    unsigned int seed
)
{
    std::ofstream out(output_file);

    if (!out){
        throw std::runtime_error("Could not open output file.");
    }

    write_european_csv_header(out);

    for (int n_paths : path_grid)
    {
        std::vector<double> prices;
        std::vector<double> abs_errors;
        std::vector<double> signed_errors;
        std::vector<double> std_errors;
        std::vector<double> ci_widths;
        std::vector<double> runtimes;

        prices.reserve(n_replications);
        abs_errors.reserve(n_replications);
        signed_errors.reserve(n_replications);
        std_errors.reserve(n_replications);
        ci_widths.reserve(n_replications);
        runtimes.reserve(n_replications);

        for (int rep = 0; rep < n_replications; rep++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            EstimatorStats stats =
                mc_price_european_call(
                    params,
                    n_paths,
                    seed + EUROPEAN_OFFSET + rep * 1000 + n_paths
                );

            auto end = std::chrono::high_resolution_clock::now();

            double runtime =
                std::chrono::duration<double>(end - start).count();

            prices.push_back(stats.price);
            abs_errors.push_back(std::abs(stats.price - EUROPEAN_REFERENCE));
            signed_errors.push_back(stats.price - EUROPEAN_REFERENCE);
            std_errors.push_back(stats.std_error);
            ci_widths.push_back(stats.ci_high - stats.ci_low);
            runtimes.push_back(runtime);
        }

        write_european_csv_row(
            out,
            n_paths,
            EUROPEAN_REFERENCE,
            mean(prices),
            std::sqrt(sample_variance(prices)),
            mean(abs_errors),
            mean(signed_errors),
            mean(std_errors),
            mean(ci_widths),
            mean(runtimes),
            median(runtimes)
        );
    }
}


/*
Run an Asian benchmark sweep and save all results to CSV.

Inputs:
    params: option parameters
    path_grid: list of path counts to test
    n_steps: number of time steps per path
    n_replications: number of repetitions per path count
    output_file: CSV output path
    seed: base random seed

Returns:
    None
*/
void run_asian_benchmark(
    const OptionParams& params,
    const std::vector<int>& path_grid,
    int n_steps,
    int n_replications,
    const std::string& output_file,
    unsigned int seed
)
{
    std::ofstream out(output_file);

    if (!out){
        throw std::runtime_error("Could not open output file.");
    }

    write_asian_csv_header(out);

    for (int n_paths : path_grid)
    {
        std::vector<double> prices;
        std::vector<double> abs_errors;
        std::vector<double> signed_errors;
        std::vector<double> std_errors;
        std::vector<double> ci_widths;
        std::vector<double> runtimes;

        prices.reserve(n_replications);
        abs_errors.reserve(n_replications);
        signed_errors.reserve(n_replications);
        std_errors.reserve(n_replications);
        ci_widths.reserve(n_replications);
        runtimes.reserve(n_replications);

        for (int rep = 0; rep < n_replications; rep++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            EstimatorStats stats =
                mc_price_asian_call(
                    params,
                    n_paths,
                    n_steps,
                    seed + ASIAN_OFFSET + rep * 1000 + n_paths
                );

            auto end = std::chrono::high_resolution_clock::now();

            double runtime =
                std::chrono::duration<double>(end - start).count();

            prices.push_back(stats.price);
            abs_errors.push_back(std::abs(stats.price - ASIAN_REFERENCE));
            signed_errors.push_back(stats.price - ASIAN_REFERENCE);
            std_errors.push_back(stats.std_error);
            ci_widths.push_back(stats.ci_high - stats.ci_low);
            runtimes.push_back(runtime);
        }

        write_asian_csv_row(
            out,
            n_steps,
            n_paths,
            ASIAN_REFERENCE,
            mean(prices),
            std::sqrt(sample_variance(prices)),
            mean(abs_errors),
            mean(signed_errors),
            mean(std_errors),
            mean(ci_widths),
            mean(runtimes),
            median(runtimes)
        );
    }
}


int main(){
    OptionParams params{
        S0,
        K,
        R,
        SIGMA,
        T
    };

    run_european_benchmark(
        params,
        PATH_GRID,
        N_REPLICATIONS,
        "results/data/cpp_european_mc_summary.csv",
        SEED
    );

    run_asian_benchmark(
        params,
        PATH_GRID,
        N_STEPS,
        N_REPLICATIONS,
        "results/data/cpp_asian_mc_summary.csv",
        SEED
    );

    return 0;
}
