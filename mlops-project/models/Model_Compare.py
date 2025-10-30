import yaml
import mlflow
import importlib
import inspect
from typing import Dict, Any
from ModelTrainer  import ModelTrainer, DataProcessor

#=============Requicitos y como usar esta clase =====================
# 1.- Importar el archivo y sus funciones y Model Trainer con las siguientes lineas
#      from MLOPS_Funcion import DataProcessor
#      from Model_compare import Experimento
# 2.- Tener el pre procesamiento de datos listos con los datos de entrenamiento requeridos para la clase DataProcessor
#       data_splits = DataProcessor(
#           X_train=X_train, y_train=y_train,
#           X_val=X_val, y_val=y_val,
#           X_test=X_test, y_test=y_test
#               )
# 3.- Crear el objeto Experimento indicando la localizacion del archivo yaml y el objeto DataProcessor
#           modelos_a_evaluar = Experimento(config_path="config.yaml",data_processor=data_splits)
#
# 4.- Entrena y evalua todos los modelos deseados de regresión en yaml con la sigueinte funcion 
#           runmodelos_a_evaluar.ejecuta_experimentos()
#
# 5.- Abre MLFlow y obten el mejor modelo del experimento 
#           Mejor_modelo = modelos_a_evaluar.mejor_modelo()


class Experimento:
    """
    Esta Clase utiliza las clases en ModelTrainer para entrenar multiples modelos de regresión y compararlos para elegir el mejor modelo 
    Con el motivo de comparar multiples modelos de importara un archivo de configuración YAML con los algoritmos a utilizar alamacenando los resultados  
    """
    
    def __init__(self, config_path: str, data_processor: DataProcessor):
        self.config_path = config_path
        self.data = data_processor
        
        # Importamos el YAML de configuración 
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.experiment_name = self.config['experiment_name']
        self.models_config = self.config['models_to_try']
        
        # Inicializamos MLflow
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow: Experimento a Correr '{self.experiment_name}'")

    def _instantiate_model(self, class_path: str) -> Any:
        """Importamos y generamos una instancia del modelo"""
        try:
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            EstimatorClass = getattr(module, class_name)
            
            # MLOps: insertamos "Random state" para reprudicir el modelo si el modelo lo soporta
            init_kwargs = {}
            if 'random_state' in inspect.signature(EstimatorClass.__init__).parameters:
                init_kwargs['random_state'] = 42
                
            return EstimatorClass(**init_kwargs)
        except Exception as e:
            print(f"Error de instancia {class_path}: {e}")
            raise

    def ejecuta_experimentos(self) -> None:
        """
        Corremos todos los modelos especificados en el config.yaml y los comparamos .
        """
        
        # Creamos el Proceso Padre para MLFLOW el cual contiene toda la ejecucion del codigo
        
        with mlflow.start_run(run_name="Comparacion de modelos ") as parent_run:
            mlflow.log_param("config_file", self.config_path)
            
            print(f"Proceso Padre Iniciado para MLFLOW: {parent_run.info.run_id}")
            print("Comenzando Evaluacion y Comparación")
            
            for model_name, model_cfg in self.models_config.items():
                
                print(f"Ejecutando modelo : {model_name} ")
                
                # Crando primer Hilo/Hijo
                with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                    
                    try:
                        # Creando instancia 
                        estimator = self._instantiate_model(model_cfg['class_path'])
                        scaler = model_cfg.get('scaler', None) # 'scaler' o None
                        
                        # Agregamos Etiquetas de MLFlow 
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_param("scaler", scaler)
                        mlflow.log_param("class_path", model_cfg['class_path'])
                        
                        # Creando objeto ModelTrainer para entrenar y evaluar los modelos 
                        trainer = ModelTrainer(
                            **self.data.__dict__, # Pasamos datos de entrenamiento 
                            estimator=estimator,
                            scaler=scaler,
                            model_name=model_name
                        )
                        
                        # Entrenando y Evaluando el modelo
                        trainer.train_model()
                        # Evaluamos el modelo entrenado
                        metrics = trainer.evaluate_model() 
                        
                        # Guardamos Resultados en MLFlow
                        mlflow.log_metric("train_rmse", metrics['train']['rmse'])
                        mlflow.log_metric("train_r2",   metrics['train']['r2'])
                        mlflow.log_metric("val_rmse",   metrics['val']['rmse'])
                        mlflow.log_metric("val_r2",     metrics['val']['r2'])
                        mlflow.log_metric("test_rmse",  metrics['test']['rmse'])
                        mlflow.log_metric("test_r2",    metrics['test']['r2'])
                        
                        #Guardamos el modelo 
                        mlflow.sklearn.log_model(trainer.pipeline, "model_pipeline")
                        
                        print(f"MLflow: {model_name} registrado (Run ID: {child_run.info.run_id})")

                    except Exception as e:
                        print(f"Error al entrenar el modelo {model_name}: {e}")
                        mlflow.set_tag("status", "FAILED")
                        mlflow.log_param("error", str(e))

            print("\nComparacion de Modelos Terminada")
            mlflow.set_tag("status", "SUCCESS")

    # -------------------------------------------------------------------
    # Funcion que obtiene el mejor modelo comparando los contenidos en el config.yaml guardados previamente en el MLFLOW 
    # -------------------------------------------------------------------
    def mejor_modelo(self) -> Dict[str, Any]:
        """
        Conecta con MLFlow para revisa que modelo se considera el mejor en la corrida del experimento de config.yaml
        """
        print("\nComparando Resultados para obtencion del mejor modelo en MLFlow")
        
        metric_to_optimize = self.config['metric_to_optimize'].replace("_", ".")
        order_mode = self.config['optimize_mode']
        
        # Obtiene el ID del experimento a evaluar
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            print("No se encontro el ID, revisar datos guardados o el ID")
            return {}
            
        # Busqueda de resultados del experimento
        best_run_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_to_optimize} {order_mode}"],
            max_results=1 # busca el mejor modelo
        )
        
        if best_run_df.empty:
            print("No se Encontraron Experimentos ejecutados")
            return {}
            
        #Exportando la información del mejor resutlado
        best_run_data = best_run_df.iloc[0]
        best_model_name = best_run_data['tags.model_name']
        best_run_id = best_run_data['run_id']
        best_metric_score = best_run_data[f"metrics.{metric_to_optimize}"]
        
        # Guarda la dirección del mejor modelo encontrado
        model_artifact_uri = f"runs:/{best_run_id}/model_pipeline"
        
        print("\n" + "="*30 + " MEJOR MODELO ENCONTRADO " + "="*30)
        print(f"Modelo:     {best_model_name}")
        print(f"Métrica:    {metric_to_optimize}")
        print(f"Valor:      {best_metric_score:.4f}")
        print(f"Run ID:     {best_run_id}")
        print(f"URI Modelo: {model_artifact_uri}")
        print("="*82)
        
        return {
            "model_name": best_model_name,
            "metric_name": metric_to_optimize,
            "score": best_metric_score,
            "run_id": best_run_id,
            "model_uri": model_artifact_uri
        }