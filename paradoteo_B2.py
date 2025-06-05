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
from paradoteo_A4 import train_nn_full_dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')


class GeneticAlgorithmWithNN:
    def __init__(self, population_size, chromosome_length, crossover_prob, mutation_prob, elitism_ratio, min_hamming_distance):
        # Αρχικοποίηση παραμέτρων ΓΑ
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
                # Δημιουργία τυχαίου χρωμοσώματος
                chromosome = [random.randint(0, 1)
                              for _ in range(self.chromosome_length)]

                # Έλεγχος κανόνων
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

    def fitness_function(self, chromosome, model, data, target, penalty_multiplier=0.05, accuracy_multiplier=0.95):
        """
        Fitness function using accuracy instead of cross-entropy loss.
        Lower fitness = better performance (minimization problem).
        """
        accuracy = GeneticAlgorithmWithNN.evaluate_nn_fixed_weights(
            model, data, target, chromosome)
        num_active_features = sum(chromosome)

        # Penalty for using more features
        penalty = (num_active_features / len(chromosome))

        # Convert accuracy to error rate for minimization
        error_rate = 1 - accuracy  # Error rate: 0 is perfect, 1 is worst

        # Combine error rate and penalty (both should be minimized)
        fitness = accuracy_multiplier * error_rate + penalty_multiplier * penalty

        return fitness

    @staticmethod
    def evaluate_nn_fixed_weights(model, data, target, chromosome):
        # Get original feature names
        original_features = data.columns.tolist()
        selected_original_features = [feature for feature, bit in zip(
            original_features, chromosome) if bit == 1]

        # Always preprocess with ALL features
        preprocessor = preprocessing_pipeline(data, chromosome=None)
        X = preprocessor.fit_transform(data)
        y = target.values

        # Get preprocessed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Create mask for preprocessed features
        mask = np.zeros(len(feature_names), dtype=bool)

        for i, feature_name in enumerate(feature_names):
            # Check if this preprocessed feature corresponds to a selected original feature
            for selected_feature in selected_original_features:
                if selected_feature in feature_name:
                    mask[i] = True
                    break

        # Apply mask
        X_masked = X.copy()
        X_masked[:, ~mask] = 0

        # Evaluate
        y_pred = model.predict(X_masked, verbose=0)
        accuracy = np.mean((y_pred > 0.5).astype(int) == y)

        return accuracy

    def evolve(self, population, model, data, target):
        # Evaluate fitness for each chromosome in the population
        fitness_scores = [self.fitness_function(
            chromosome, model, data, target) for chromosome in population]

        # Perform elitism: Retain the best individuals
        num_elites = int(self.elitism_ratio * self.population_size)
        elites = sorted(zip(population, fitness_scores),
                        key=lambda x: x[1])[:num_elites]
        next_population = [elite[0]
                           for elite in elites]  # Retain elite chromosomes

        # Generate new individuals through crossover and mutation
        while len(next_population) < self.population_size:
            # Select two parents randomly
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # Perform uniform crossover
            offspring = [
                parent1[i] if random.random(
                ) < self.crossover_prob else parent2[i]
                for i in range(self.chromosome_length)
            ]

            # Perform mutation
            for i in range(self.chromosome_length):
                if random.random() < self.mutation_prob:
                    # Flip the bit (0 → 1, 1 → 0)
                    offspring[i] = 1 - offspring[i]

            next_population.append(offspring)

        return next_population

    def run_fixed_weights_multiple_runs(self, runs, max_generations, data, target, hidden_units, learning_rate, momentum, l2_lambda, patience=10, min_improvement=0.01):
        """
        Εκτέλεση του ΓΑ για πολλαπλά τρεξίματα και υπολογισμός μέσου όρου της καλύτερης τιμής fitness ανά γενιά.
        """
        # Train the NN once with the full dataset
        model = train_nn_full_dataset()

        all_best_fitness = []  # Αποθήκευση της καλύτερης τιμής fitness ανά γενιά για κάθε τρέξιμο
        # Track the actual number of generations for each run
        actual_generations_per_run = []
        best_chromosome_overall = None  # Track the best chromosome across all runs
        # Track the best fitness across all runs
        best_fitness_overall = float('inf')

        for run in range(runs):
            print(f"Run {run + 1}/{runs}")

            # Initialize population for this run
            population = self.initialize_population()
            best_fitness_per_generation = []
            no_generation_improvement_counter = 0
            previous_best_fitness = float('inf')

            for generation in range(max_generations):
                # Evaluate fitness for each chromosome using the fixed weights
                fitness_scores = [self.fitness_function(
                    chromosome, model, data, target) for chromosome in population]
                # Get the best fitness (minimization)
                best_fitness = min(fitness_scores)
                best_fitness_per_generation.append(best_fitness)

                # Track the best chromosome across all runs
                best_chromosome_index = fitness_scores.index(best_fitness)
                best_chromosome = population[best_chromosome_index]
                if best_fitness < best_fitness_overall:
                    best_fitness_overall = best_fitness
                    best_chromosome_overall = best_chromosome

                # Check improvement
                improvement = previous_best_fitness - best_fitness
                if improvement < min_improvement:
                    no_generation_improvement_counter += 1
                else:
                    no_generation_improvement_counter = 0

                previous_best_fitness = best_fitness

                # Termination criteria
                if no_generation_improvement_counter >= patience:
                    print(
                        f"Τερματισμός: Το καλύτερο άτομο δεν βελτιώθηκε για {patience} γενιές.")
                    # Track the actual number of generations
                    actual_generations_per_run.append(generation + 1)
                    break
                if generation >= max_generations - 1:
                    print(
                        f"Τερματισμός: Έχει ξεπεραστεί ο μέγιστος αριθμός γενεών ({max_generations}).")
                    # Track the actual number of generations
                    actual_generations_per_run.append(max_generations)
                    break

                # Evolve population
                population = self.evolve(population, model, data, target)

            # Store the best fitness values for this run
            all_best_fitness.append(best_fitness_per_generation)

        # Compute average best fitness across runs
        max_actual_generations = max(actual_generations_per_run)
        padded_fitness = []
        for fitness_list in all_best_fitness:
            if len(fitness_list) < max_actual_generations:
                # Pad with the last fitness value
                last_value = fitness_list[-1]
                padded_list = fitness_list + \
                    [last_value] * (max_actual_generations - len(fitness_list))
                padded_fitness.append(padded_list)
            else:
                padded_fitness.append(fitness_list)
        avg_best_fitness = np.mean(all_best_fitness, axis=0)
        # Extract selected features from the best chromosome
        selected_features = [feature for feature, bit in zip(
            data.columns, best_chromosome_overall) if bit == 1]

        print(f"\nBest Chromosome Overall: {best_chromosome_overall}")
        print(f"Best Fitness Overall: {best_fitness_overall}")
        print(f"Selected Features: {selected_features}")

        return avg_best_fitness, max_actual_generations, best_chromosome_overall, selected_features


def plot_evolution_curve(avg_best_fitness, generations, test_case):
    """
    Σχεδίαση της καμπύλης εξέλιξης (απόδοση/αριθμός γενεών).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 1), avg_best_fitness,
             marker='o', label='Average Best Fitness')
    plt.title(
        f"Evolution Curve (Population: {test_case['population_size']}, Crossover: {test_case['crossover_prob']}, Mutation: {test_case['mutation_prob']})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()

    # Dynamically name the PNG file
    filename = f"evolution_curve_pop_{test_case['population_size']}_cross_{test_case['crossover_prob']}_mut_{test_case['mutation_prob']}.png"
    plt.savefig(filename)
    plt.show()


def preprocessing_pipeline(data, chromosome):
    data = data.copy()
    # Convert all numeric columns to float32
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].astype('float32')

    # Convert integer categorical columns to categorical type
    data['EducationLevel'] = data['EducationLevel'].astype("category")
    data['Ethnicity'] = data['Ethnicity'].astype("category")

    # Feature selection based on chrosome, if its not provided use all features
    if chromosome is not None:
        selected_features = [feature for feature, bit in zip(
            data.columns, chromosome) if bit == 1]
    else:
        selected_features = data.columns

    # Define preprocessing rules as a dictionary
    preprocessing_rules = {
        'normalize': ['AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'ADL'],
        'standardize': ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                        'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment'],
        'encode': ['EducationLevel', 'Ethnicity']
    }

    # Filter preprocessing rules based on selected features
    columns_for_normalization = [
        col for col in preprocessing_rules['normalize'] if col in selected_features]
    columns_for_standardization = [
        col for col in preprocessing_rules['standardize'] if col in selected_features]
    columns_for_encoding = [
        col for col in preprocessing_rules['encode'] if col in selected_features]

    # Pipeline for normalization
    normalization_pipeline = Pipeline([
        ('normalizer', MinMaxScaler(feature_range=(0, 1)))  # Normalization
    ])

    # Pipeline for standardization
    standardization_pipeline = Pipeline([
        ('scaler', StandardScaler())  # Standardization
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        # One-hot encoding
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('normalize', normalization_pipeline, columns_for_normalization),
        ('standardize', standardization_pipeline, columns_for_standardization),
        ('categorical', categorical_pipeline, columns_for_encoding)
    ])

    return preprocessor


if __name__ == "__main__":
    # Φόρτωση δεδομένων
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    df = pd.read_csv("alzheimers_disease_data.csv")
    target_column = df['Diagnosis']
    df = df.drop(columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])

    # Υπερπαράμετροι ΝΝ
    hidden_units = 76
    learning_rate = 0.05
    momentum = 0.6
    l2_lambda = 0.0001

    # Υπερπαράμετροι ΓΑ
    chromosome_length = df.shape[1]
    max_generations = 100
    patience = 10
    min_improvement = 0.01
    runs = 10  # Αριθμός τρεξιμάτων

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

    # Results storage
    results = []
    global_best_chromosome = None
    global_best_fitness = float('inf')
    global_selected_features = None
    for idx, test_case in enumerate(test_cases, start=1):
        print(f"\nRunning Test Case {idx}: {test_case}")

        # Δημιουργία ΓΑ
        ga = GeneticAlgorithmWithNN(
            population_size=test_case["population_size"],
            chromosome_length=chromosome_length,
            crossover_prob=test_case["crossover_prob"],
            mutation_prob=test_case["mutation_prob"],
            elitism_ratio=0.1,  # Fixed elitism ratio
            min_hamming_distance=2
        )

        # Εκτέλεση ΓΑ για πολλαπλά τρεξίματα
        avg_best_fitness, generations, best_chromosome, selected_features = ga.run_fixed_weights_multiple_runs(
            runs, max_generations, df, target_column, hidden_units, learning_rate, momentum, l2_lambda, patience, min_improvement
        )

        if avg_best_fitness[-1] < global_best_fitness:
            global_best_fitness = avg_best_fitness[-1]
            global_best_chromosome = best_chromosome
            global_selected_features = selected_features

        # Σχεδίαση καμπύλης εξέλιξης
        plot_evolution_curve(avg_best_fitness, generations, test_case)

        # Αποθήκευση αποτελεσμάτων
        results.append({
            "Test Case": idx,
            "Population Size": test_case["population_size"],
            "Crossover Probability": test_case["crossover_prob"],
            "Mutation Probability": test_case["mutation_prob"],
            # Final fitness value
            "Average Best Fitness": avg_best_fitness[-1],
            "Average Generations": len(avg_best_fitness)  # Total generations
        })

    # Print results
    print(
        f"\nGlobal Best Chromosome:{global_best_chromosome} and best fitness: {global_best_fitness}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("ga_results.csv", index=False)
    print("\nResults:")
    for result in results:
        print(result)
