import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs from TensorFlow

from absl import logging
logging.set_verbosity(logging.ERROR)




from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygad
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sys import stdout
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import time
from sklearn.svm import SVR
df = pd.read_excel("datasets/Dataset_2.xlsx",sheet_name="preproc")
import tensorflow as tf
from keras.callbacks import EarlyStopping
import gc
import psutil
from multiprocessing import Pool, cpu_count, get_context
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


trials = 10
random_cv = np.random.seed(777)
seeds = np.random.choice(trials + 1, size=trials, replace= False)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'




#INPUT DATA
vip_bands = []
with open(f"VIPs_350_Fixed_25_Bands.txt","r") as f:
    vip_bands = f.read().splitlines()

vip_bands = [float(vip_band) for vip_band in vip_bands]
print(vip_bands)

X = (df.loc[:,vip_bands]).values
print(X.shape)

#OUTPUTDATA
y = (df.iloc[:,7]).values
print(y.shape)



#NEGNRMEPiqr
def NEGNRMSEPiqr(observed_values, predicted_values):
    # Calculate RMSEP
    rmsep = np.sqrt(np.mean((observed_values - predicted_values) ** 2))
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(observed_values, 25)
    Q3 = np.percentile(observed_values, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    return -rmsep/IQR


NEGNRMSEPiqrscorer = make_scorer(NEGNRMSEPiqr)



#NRMEPiqr
def NRMSEPiqr(observed_values, predicted_values):
    # Calculate RMSEP
    rmsep = np.sqrt(np.mean((observed_values - predicted_values) ** 2))
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(observed_values, 25)
    Q3 = np.percentile(observed_values, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    return rmsep/IQR


NRMSEPiqrscorer = make_scorer(NRMSEPiqr)




def build_nn(input_size, filters=32, kernel_size=3, dropout_rate=0.2, learning_rate=0.001):

        '''
        inputs = tf.keras.Input(shape=(self._m,))
        x = tf.keras.layers.Reshape((self._m, 1))(inputs)
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model
        '''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        model.add(tf.keras.layers.Reshape((input_size, 1)))
        # Convolutional layers
        model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.06), activation='elu'))
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.06), activation='elu'))
        # Fully connected layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='elu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model



def train_and_evaluate_nn(args):
        X_train, X_test, Y_train, Y_test = args
        model = build_nn(input_size=X_train.shape[1])   

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,       
            mode='auto',         
            patience=50,
            restore_best_weights=True, 
        )
        
        model.fit(X_train, Y_train, epochs=500, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])
        Y_pred = model.predict(X_test, verbose=0).flatten()
        fitness = NEGNRMSEPiqr(Y_test, Y_pred)
        tf.keras.backend.clear_session()
        gc.collect()
        return fitness





def fitness_function(ga_instance, solution, solution_idx):

    epsilon=0.3178
    C = 5164
    alpha = 0.00122457013067159
    NNRMSE = []
    # Select wavelengths based on the solution
    selected_wavelengths = np.where(solution == 1)[0]
    
    if len(selected_wavelengths) == 0:
        return -1e6
    X_selected = X[:, selected_wavelengths]

    data_list = []
    for i in range(trials):
        X_train, X_test, Y_train, Y_test = train_test_split(X_selected, y, test_size=0.25, random_state=seeds[i])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        data_list.append((X_train_scaled, X_test_scaled, Y_train, Y_test))

    num_workers = max(1, cpu_count() - 2)
    with get_context("fork").Pool(processes=num_workers) as pool:
        try:
            results = pool.map(train_and_evaluate_nn, data_list)
        except Exception as e:
            print(f"Multiprocessing error: {e}")
            return -1e6

    return np.mean(results)

   



# Number of wavelengths (genes)
num_wavelengths = X.shape[1]

# Create the GA instance
ga_instance = pygad.GA(
    num_generations=20,             # Number of generations
    num_parents_mating=20,          # Number of parents for crossover
    keep_elitism=6,
    fitness_func=fitness_function,  # Fitness function
    sol_per_pop=40,                 # Population size
    num_genes=num_wavelengths,      # Number of wavelengths
    gene_type=int,                  # Binary genes (0 or 1)
    init_range_low=0,               # Genes initialized to 0
    init_range_high=2,              # Genes initialized to 1 (binary)
    parent_selection_type="sss",    # Stochasting sampling selection
    crossover_type="single_point",  # Single-point crossover
    mutation_type="random",         # Random mutation
    mutation_percent_genes= 1,  # Mutate 1% of the genes
    on_generation=lambda ga: print(f"Generation: {ga.generations_completed}, Fitness: {ga.best_solution()[1]}")
)

# Run the genetic algorithm
start_time = time.time()
ga_instance.run()
final_time = (time.time() -start_time)/60

print("Execution time (min): " + str(final_time))

# Best solution
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Selected wavelengths: {np.where(solution == 1)[0]}")
print(f"Best fitness value: {solution_fitness}")

# Plot the fitness values over generations
ga_instance.plot_fitness()













def NRMSEPiqr(observed_values, predicted_values):
    # Calculate RMSEP
    rmsep = np.sqrt(np.mean((observed_values - predicted_values) ** 2))
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(observed_values, 25)
    Q3 = np.percentile(observed_values, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    return rmsep/IQR


NRMSEPiqrscorer = make_scorer(NRMSEPiqr)


trials = 100
random_cv = np.random.seed(777)
seeds = np.random.choice(trials + 1, size=trials, replace= False)
ripartition = 0.25
final_results_pls = dict()


result_NRMSEP_CNN = dict()
NRMSE = []
total_time = 0

epsilon=0.3178
C = 5164
alpha = 0.00122457013067159
for i in range(trials):
    X_train, X_test, Y_train, Y_test = train_test_split(X[:, np.where(solution == 1)[0]], y, test_size=ripartition, random_state=seeds[i])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = build_nn(input_size=X_train_scaled.shape[1])

    early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,       
            mode='auto',         
            patience=50,
            restore_best_weights=True, 
    )
    start_time = time.time()
    model.fit(X_train_scaled, Y_train, epochs=500, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])
    elapsed_time = time.time() - start_time
        
    Y_pred_nn = model.predict(X_test_scaled, verbose=0).flatten()
    NRMSE.append(NRMSEPiqr(Y_test, Y_pred_nn))
    total_time += elapsed_time
    del model
    


result_NRMSEP_CNN["Test size(%)"] =  str(ripartition *100 )
result_NRMSEP_CNN["Mean NRMSEPiqr"] = np.mean(NRMSE)
result_NRMSEP_CNN["Var NRMSEPiqr"] = np.var(NRMSE)
result_NRMSEP_CNN["Mean Training Time (sec)"] = total_time/trials
       


vip_np = np.array(vip_bands)
vip_np[np.where(solution == 1)[0]]



waves = np.array(vip_bands)
waves = waves[np.where(solution == 1)[0]]

with open(f"GA_VIP350_CNN_Bands.txt","w") as f:
    f.write("Mean NRMSEPiqr:"+str(result_NRMSEP_CNN["Mean NRMSEPiqr"]) + 
            " , "+ "Var NRMSEPiqr: "+str(result_NRMSEP_CNN["Var NRMSEPiqr"]) + " , " + 
            "Mean Training Time (sec): "+str(result_NRMSEP_CNN["Mean Training Time (sec)"]))
    f.write("\n")
    for wave in waves:
        f.write(str(wave)+"\n")
