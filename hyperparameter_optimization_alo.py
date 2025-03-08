import argparse
import random
import uuid
import json
from train import train_model


learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
stochastic_depth_values = [0.5, 0.6, 0.7, 0.8, 0.9]
bounds = [learning_rates, stochastic_depth_values, stochastic_depth_values]

def get_objective_function(train_data_path, train_labels_path, val_data_path, val_labels_path, num_epochs):
    """
    Objective function: Calculates the best validation accuracy by running train_model with the given hyperparameters.
    """
    def objective(hyperparams):
        lr, sd1, sd2 = hyperparams
        model_name = "resnet_sd"
        checkpoint_name = "temp_" + str(uuid.uuid4())
        batch_size = 32  # ثابت
        best_val_acc = train_model(model_name, checkpoint_name,
                                   train_data_path, train_labels_path,
                                   val_data_path, val_labels_path,
                                   num_epochs, lr, batch_size, sd1, sd2)
        return 1 - best_val_acc
    return objective


def alo_optimization(objective_function, bounds, num_agents=5, max_iter=3):
    dim = len(bounds)
    antlions = [[random.choice(bounds[d]) for d in range(dim)] for _ in range(num_agents)]
    ants = antlions[:]
    elite = min(antlions, key=objective_function)
    elite_score = objective_function(elite)
    
    for iter in range(max_iter):
        for i in range(num_agents):
            new_pos = [random.choice(bounds[d]) for d in range(dim)]
            ants[i] = new_pos
            if objective_function(ants[i]) < objective_function(antlions[i]):
                antlions[i] = ants[i][:]
        
        current_elite = min(antlions, key=objective_function)
        current_score = objective_function(current_elite)
        if current_score < elite_score:
            elite = current_elite[:]
            elite_score = current_score
        
        print(f"ALO iter {iter+1}/{max_iter}, best score: {elite_score}")
    
    return elite, elite_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to training data tensor')
    parser.add_argument('--train_labels_path', type=str, required=True, help='Path to training labels tensor')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to validation data tensor')
    parser.add_argument('--val_labels_path', type=str, required=True, help='Path to validation labels tensor')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for evaluation during optimization')
    args = parser.parse_args()

    objective = get_objective_function(args.train_data_path, args.train_labels_path,
                                       args.val_data_path, args.val_labels_path,
                                       args.num_epochs)

    print("Starting ALO Optimization")
    best_alo, score_alo = alo_optimization(objective, bounds, num_agents=5, max_iter=3)

    results = {
        "ALO_best_hyperparameters": best_alo,
        "ALO_best_score": score_alo
    }

    with open("best_hyperparameters.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nOptimization Results:")
    print(f"ALO best hyperparameters: {best_alo} with objective score: {score_alo}")
