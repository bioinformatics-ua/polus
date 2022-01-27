from polus.training import ClassifierTrainer
from polus.metrics import MacroF1Score
from polus.callbacks import LossSmoothCallback, ValidationDataCallback, ConsoleLogCallback, TimerCallback, EarlyStop
from polus.data import DataLoader
from polus.models import SequentialPolusClassifier
from polus.hpo import HPO_Objective, parameter

import tensorflow as tf




def init_trainer():    
    ## data
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    def train_gen(data_x, data_y):
        
        def generator():
            
            for i in range(len(data_x)):
                yield {"x":data_x[i], "y":data_y[i]}
            
        return generator
        
    
    def normalize_img(data):
        return tf.cast(data["x"], tf.float32) / 255., tf.cast(data["y"], tf.int32)
    
    
    ds_train = DataLoader(train_gen(x_train, y_train))
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(len(x_train))
    ds_train = ds_train.batch(128, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    ds_test = DataLoader(train_gen(x_test, y_test))
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = SequentialPolusClassifier([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(parameter(128, lambda trial:trial.suggest_int('dense_units', 32, 512)), activation=parameter('relu', lambda trial:trial.suggest_categorical('dense_activation', ["relu", "selu", "tanh"]))),
      tf.keras.layers.Dense(10)
    ])
    
    #def model_inference(model, sample):
    #    return tf.argmax(model(sample[0]), axis=-1, output_type=tf.int32), sample[1]
       # return sample
    
    optimizer = tf.keras.optimizers.Adam(parameter(0.001, lambda trial: trial.suggest_loguniform("lr", 1e-6, 1e-3)))
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    trainer = ClassifierTrainer(model,
                                optimizer,
                                loss,
                                metrics=[MacroF1Score(num_classes=10)])
    
    callbacks = [LossSmoothCallback(output=True), # if output is True the smooth should be positioned before all the streaming outputs
                 TimerCallback(), # This callback should be positioned before all the streaming outputs
                 ValidationDataCallback(ds_test, name="MNIST_Test"),
                 ConsoleLogCallback(), # Prints the training on the console
                 EarlyStop(),
                ]
    
    # this set all the parameters that are used to call
    # trainer.train
    trainer.changing_train_config(tf_dataset=ds_train, 
                                  epochs=5,
                                  callbacks=callbacks)

    return trainer




if __name__ == "__main__":
    
    with HPO_Objective(init_trainer, "MNIST_Test", "MacroF1Score") as obj:
        print("HPO backend", obj.backend)
        
        print("Optuna config")
        for k,v in obj.optuna_cfg.items():
            print("\t",k,":",v)
        
        # change the default cfg
        obj.change_optuna_config(n_trials=10)
    
    print(obj.study.best_trial)
    # if we want to access the study just do
    # obj.study
