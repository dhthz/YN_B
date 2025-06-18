import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Input
from keras.src.optimizers import SGD
from keras.src.regularizers import L2
import matplotlib.pyplot as plt
import os
import warnings
from paradoteo_A4 import train_NN_full_dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')


class GeneticAlgorithmWithNN:
    def __init__(self, population_size, chromosome_length, crossover_prob, mutation_prob, elitism_ratio, min_hamming_distance):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_ratio = elitism_ratio
        self.min_hamming_distance = min_hamming_distance  # Ελάχιστη απόσταση Hamming

    # Helper function to find the Hamming distance between two chromosomes
    @staticmethod
    def hamming_distance(chromosome1, chromosome2):
        return sum(bit1 != bit2 for bit1, bit2 in zip(chromosome1, chromosome2))

    def initialize_population(self):
        population = []
        max_attempts = 1000  # Prevent infinite loops

        for _ in range(self.population_size):
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                chromosome = [random.randint(0, 1)
                              for _ in range(self.chromosome_length)]

                num_ones = sum(chromosome)
                if num_ones >= 2 and num_ones < self.chromosome_length:
                    if len(population) == 0 or all(GeneticAlgorithmWithNN.hamming_distance(chromosome, existing) >= self.min_hamming_distance for existing in population):
                        population.append(chromosome)
                        break

            if attempts >= max_attempts:
                print(
                    f"Warning: Could not find valid chromosome after {max_attempts} attempts")
                # Add a random valid chromosome as fallback
                fallback = [0] * self.chromosome_length
                fallback[0] = 1  # Ensure at least 2 ones
                fallback[1] = 1
                population.append(fallback)

        return population

    def fitness_function(self, chromosome, model, data, target, original_columns, penalty_weight=0.05, accuracy_weight=0.95):
        accuracy = GeneticAlgorithmWithNN.evaluate_nn_fixed_weights(
            model, data, target, chromosome, original_columns)
        num_active_features = sum(chromosome)

        # Penalty for using more features (0 to 1 range)
        penalty = (num_active_features / len(chromosome)) * penalty_weight

        # Weighted accuracy minus penalty (higher = better)
        fitness = accuracy_weight * accuracy - penalty

        return fitness

    @staticmethod
    def evaluate_nn_fixed_weights(model, processed_data, target, chromosome, original_columns):
        processed_features = [col for col in processed_data.columns
                              if col not in ['PatientID', 'Diagnosis', 'DoctorInCharge']]

        X = processed_data[processed_features].values.astype('float32')
        y = target.values

        # Create mapping from original columns to processed columns
        mapping, _ = create_original_to_processed_mapping(
            original_columns, processed_data)

        # Apply chromosome selection
        X_masked = X.copy()

        # For each chromosome bit
        for chromosome_idx, bit in enumerate(chromosome):
            if bit == 0 and chromosome_idx in mapping:
                # If this original feature is not selected, zero out all corresponding processed columns
                processed_column_indices = mapping[chromosome_idx]
                for col_idx in processed_column_indices:
                    X_masked[:, col_idx] = 0

        # Evaluate
        y_pred = model.predict(X_masked, verbose=0)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        accuracy = np.mean((y_pred > 0.5).astype(int) == y)
        return accuracy

    def evolve(self, population, model, data, target, original_columns):
        # Evaluate fitness for each chromosome
        fitness_scores = [self.fitness_function(
            chromosome, model, data, target, original_columns) for chromosome in population]

        # Perform elitism: Retain the best individuals (HIGHEST fitness)
        num_elites = int(self.elitism_ratio * self.population_size)
        elites = sorted(zip(population, fitness_scores),
                        # REVERSE=True for maximization
                        key=lambda x: x[1], reverse=True)[:num_elites]
        next_population = [elite[0] for elite in elites]

        # Generate new individuals through crossover and mutation
        while len(next_population) < self.population_size:
            parent1 = self.tournament_selection(
                population, fitness_scores, tournament_size=3)
            parent2 = self.tournament_selection(
                population, fitness_scores, tournament_size=3)

            offspring = [
                parent1[i] if random.random(
                ) < self.crossover_prob else parent2[i]
                for i in range(self.chromosome_length)
            ]

            for i in range(self.chromosome_length):
                if random.random() < self.mutation_prob:
                    offspring[i] = 1 - offspring[i]

            next_population.append(offspring)

        return next_population

    @staticmethod
    def tournament_selection(population, fitness_scores, tournament_size=3):
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(
            range(len(population)), tournament_size)

        # Find the best fitness in tournament (higher = better for maximization)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])

        return population[best_idx]

    def run_fixed_weights_multiple_runs(self, runs, max_generations, processed_data, target, original_columns, patience=10, min_improvement=0.01):

        model = train_NN_full_dataset()

        all_best_fitness_per_run = []
        all_best_fitness_evolution = []
        actual_generations_per_run = []
        best_chromosome_overall = None
        best_fitness_overall = float('-inf')

        print(f"    Doing {runs} runs:")

        for run in range(runs):
            print(f"      Run {run + 1}/{runs}...", end="")

            population = self.initialize_population()
            best_fitness_per_generation = []
            no_generation_improvement_counter = 0
            previous_best_fitness = float('-inf')

            for generation in range(max_generations):
                fitness_scores = [self.fitness_function(
                    chromosome, model, processed_data, target, original_columns) for chromosome in population]

                best_fitness_this_generation = max(fitness_scores)
                best_fitness_per_generation.append(
                    best_fitness_this_generation)

                best_chromosome_index = fitness_scores.index(
                    best_fitness_this_generation)
                best_chromosome = population[best_chromosome_index]

                if best_fitness_this_generation > best_fitness_overall:
                    best_fitness_overall = best_fitness_this_generation
                    best_chromosome_overall = best_chromosome

                improvement = best_fitness_this_generation - previous_best_fitness
                if improvement < min_improvement:
                    no_generation_improvement_counter += 1
                else:
                    no_generation_improvement_counter = 0

                previous_best_fitness = best_fitness_this_generation

                # Early stopping
                if no_generation_improvement_counter >= patience:
                    actual_generations_per_run.append(generation + 1)
                    break
                elif generation >= max_generations - 1:
                    actual_generations_per_run.append(max_generations)
                    break

                population = self.evolve(
                    population, model, processed_data, target, original_columns)

            all_best_fitness_evolution.append(
                best_fitness_per_generation)
            all_best_fitness_per_run.append(
                best_fitness_this_generation)
            print(
                f" Finished in {len(best_fitness_per_generation)} generations")

        max_actual_generations = max(actual_generations_per_run)

        padded_best_fitness = []
        for fitness_list in all_best_fitness_evolution:
            if len(fitness_list) < max_actual_generations:
                last_value = fitness_list[-1]
                padded_list = fitness_list + \
                    [last_value] * (max_actual_generations - len(fitness_list))
                padded_best_fitness.append(padded_list)
            else:
                padded_best_fitness.append(fitness_list)

        avg_best_fitness_evolution = np.mean(padded_best_fitness, axis=0)

        avg_best_fitness_final = np.mean(all_best_fitness_per_run)

        selected_features = [feature for feature, bit in zip(
            original_columns, best_chromosome_overall) if bit == 1]
        num_selected_features = len(selected_features)

        print(
            f"    Finished: {num_selected_features} features were selected")
        print(
            f"    Average Best Fitness from  {runs} runs: {avg_best_fitness_final:.4f}")

        return {
            # for plotting
            'avg_best_fitness_evolution': avg_best_fitness_evolution,
            'avg_best_fitness_final': avg_best_fitness_final,
            'generations': max_actual_generations,
            'best_chromosome': best_chromosome_overall,
            'selected_features': selected_features,
            'num_features': num_selected_features,
            'best_fitness_overall': best_fitness_overall,
            'all_runs_fitness': all_best_fitness_per_run
        }


def create_original_to_processed_mapping(original_columns, processed_df):
    # Get processed feature columns (exclude non-features)
    processed_features = [col for col in processed_df.columns
                          if col not in ['PatientID', 'Diagnosis', 'DoctorInCharge']]
    mapping = {}
    for i, original_col in enumerate(original_columns):
        # chromosome index -> list of processed column indices
        mapping[i] = []
        if original_col in ['EducationLevel', 'Ethnicity']:
            # Find all processed columns that start with this name
            for j, processed_col in enumerate(processed_features):
                if processed_col.startswith(original_col + '_'):
                    mapping[i].append(j)
        else:
            # Direct mapping for regular features
            if original_col in processed_features:
                j = processed_features.index(original_col)
                mapping[i].append(j)
    return mapping, processed_features


def plot_evolution_curve(results, test_case):
    plt.figure(figsize=(12, 8))

    x = np.array(range(1, results['generations'] + 1))
    y = results['avg_best_fitness_evolution']

    if len(x) >= 4:
        try:
            from scipy.interpolate import make_interp_spline
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)

            plt.plot(x_smooth, y_smooth, linewidth=4, color='blue', alpha=0.8,
                     label=f'Avg Best Fitness')
            plt.scatter(x, y, color='darkblue', s=50, zorder=5, alpha=0.8)
        except ImportError:
            plt.plot(x, y, 'o-', linewidth=3, markersize=8, color='blue',
                     label=f'Avg Best Fitness')
    else:
        plt.plot(x, y, 'o-', linewidth=3, markersize=10, color='blue',
                 label=f'Avg Best Fitness')

    # Formatting
    plt.title(f"Evolution Curve (10 Runs)\n"
              f"Population: {test_case['population_size']}, Crossover: {test_case['crossover_prob']}, "
              f"Mutation: {test_case['mutation_prob']}", fontsize=14, fontweight='bold')

    plt.xlabel("Number of generations", fontsize=12)
    plt.ylabel("Avg Best Fitness", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    avg_final = results['avg_best_fitness_final']

    stats_text = f'Generations: {results["generations"]}\n'
    stats_text += f'Starting Fitness: {y[0]:.4f}\n'
    stats_text += f'Final Fitness: {y[-1]:.4f}\n'
    stats_text += f'Average (10 runs): {avg_final:.4f}\n'
    stats_text += f'Characteristics: {results["num_features"]}\n'
    stats_text += f'Selected: {", ".join(results["selected_features"][:3])}{"..." if len(results["selected_features"]) > 3 else ""}'

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10)

    plt.tight_layout()

    filename = f"evolution_avg_pop_{test_case['population_size']}_cross_{test_case['crossover_prob']}_mut_{test_case['mutation_prob']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    original_df = pd.read_csv("alzheimers_disease_data.csv")
    original_columns = [col for col in original_df.columns
                        if col not in ['PatientID', 'Diagnosis', 'DoctorInCharge']]

    processed_df = pd.read_csv("processed_data.csv")
    target_column = processed_df['Diagnosis']
    processed_df = processed_df.drop(
        columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
    original_df = original_df.drop(
        columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])

    # GA hyperparameters
    chromosome_length = original_df.shape[1]
    max_generations = 100
    patience = 10
    min_improvement = 0.01
    runs = 10

    test_cases = [
        {"population_size": 20, "crossover_prob": 0.6, "mutation_prob": 0.00},
        {"population_size": 20, "crossover_prob": 0.6, "mutation_prob": 0.01},
        {"population_size": 20, "crossover_prob": 0.6, "mutation_prob": 0.10},
        {"population_size": 20, "crossover_prob": 0.9, "mutation_prob": 0.01},
        {"population_size": 20, "crossover_prob": 0.1, "mutation_prob": 0.01},
        {"population_size": 200, "crossover_prob": 0.6, "mutation_prob": 0.00},
        {"population_size": 200, "crossover_prob": 0.6, "mutation_prob": 0.01},
        {"population_size": 200, "crossover_prob": 0.6, "mutation_prob": 0.10},
        {"population_size": 200, "crossover_prob": 0.9, "mutation_prob": 0.01},
        {"population_size": 200, "crossover_prob": 0.1, "mutation_prob": 0.01},
    ]

    global_best_fitness = float('-inf')
    global_best_test_case = None
    global_best_chromosome = None
    global_selected_features = None

    for idx, test_case in enumerate(test_cases, start=1):
        print(f"\n{'='*60}")
        print(f"Test Case {idx}/{len(test_cases)}: Pop={test_case['population_size']}, "
              f"Cross={test_case['crossover_prob']}, Mut={test_case['mutation_prob']}")
        print(f"{'='*60}")

        # Create and run GA
        ga = GeneticAlgorithmWithNN(
            population_size=test_case["population_size"],
            chromosome_length=chromosome_length,
            crossover_prob=test_case["crossover_prob"],
            mutation_prob=test_case["mutation_prob"],
            elitism_ratio=0.1,
            min_hamming_distance=2
        )

        results = ga.run_fixed_weights_multiple_runs(
            runs, max_generations, processed_df, target_column, original_columns,
            patience, min_improvement
        )

        # Plots after 10 runs
        plot_evolution_curve(results, test_case)

        # Store results using the AVERAGE fitness from 10 runs
        estimated_accuracy = (results['avg_best_fitness_final'] + 0.05) / 0.95
        feature_reduction_percent = (
            1 - results['num_features'] / len(original_columns)) * 100

        csv_row = {
            'Test_Case_ID': idx,
            'Population_Size': test_case["population_size"],
            'Crossover_Prob': test_case["crossover_prob"],
            'Mutation_Prob': test_case["mutation_prob"],
            'Avg_Best_Fitness': results['avg_best_fitness_final'],
            'Best_Fitness_Overall': results['best_fitness_overall'],
            'Estimated_Accuracy': f"{estimated_accuracy:.1%}",
            'Features_Selected': results['num_features'],
            'Total_Features': len(original_columns),
            'Feature_Reduction': f"{feature_reduction_percent:.1%}",
            'Generations': results['generations'],
            'Selected_Features': '|'.join(results['selected_features'])
        }
        # Update global best using AVERAGE fitness
        if results['avg_best_fitness_final'] > global_best_fitness:
            global_best_fitness = results['avg_best_fitness_final']
            global_best_test_case = idx
            global_best_chromosome = results['best_chromosome']
            global_selected_features = results['selected_features']

        # Quick status
        print(f"  Avg Fitness (10 runs): {results['avg_best_fitness_final']:.4f}, "
              f"Best Overall: {results['best_fitness_overall']:.4f}, "
              f"Features: {results['num_features']}")

    # Trains the NN with the selected columns from the GA
    train_NN_full_dataset(mode=1)
