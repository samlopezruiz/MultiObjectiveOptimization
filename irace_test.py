from irace_main import opt_func
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print(opt_func(epsilon_archive=0.05,
                   increase_pop_factor=3,
                   global_steps=5,
                   n_parallel_gen=20))
