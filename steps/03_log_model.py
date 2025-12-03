import os
import mlflow
from mlflow import MlflowClient
import datetime as dt
import json

FIXED_MODEL_NAME = "yolo-rock-paper-scissors"

def setup_mlflow_environment():
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9444'
    os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAIOSFODNN7EXAMPLE'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
    mlflow.set_tracking_uri("http://localhost:5000")

def load_training_info(info_file="training_info.json"):
    try:
        with open(info_file, 'r') as f:
            training_info = json.load(f)
        return training_info
    except FileNotFoundError:
        print(f"Training info file not found: {info_file}")
        return None
    except Exception as e:
        print(f"Error loading training info: {e}")
        return None

def get_or_create_model(client, model_name):
    """Get existing model or create new one if it doesn't exist"""
    try:
        model = client.get_registered_model(model_name)
        print(f"Using existing model: {model_name}")
        
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            print(f"Found {len(versions)} existing versions:")
            for version in sorted(versions, key=lambda x: int(x.version)):
                stage_tag = version.tags.get("stage", "None") if version.tags else "None"
                print(f"   └── Version {version.version} (Stage: {stage_tag})")
        
        return model
        
    except Exception as e:
        print(f"New model '{model_name}' not found, creating new model...")
        try:
            model = client.create_registered_model(
                name=model_name,
                description=f"YOLO model for Rock-Paper-Scissors detection - Optimized with Optuna"
            )
            print(f"Model '{model_name}' created successfully")
            return model
        except Exception as create_error:
            print(f"Error creating model: {create_error}")
            return None

def get_next_version(client, model_name):
    """Get the next version number for the model"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return "1"
        
        max_version = max([int(v.version) for v in versions])
        next_version = str(max_version + 1)
        print(f"🔢 Next version will be: {next_version}")
        return next_version
        
    except Exception as e:
        print(f"Error determining next version: {e}")
        return "1"

def update_training_info(training_info, model_name, model_version, run_id, info_file="training_info.json"):
    """Update training_info.json with MLflow model information"""
    training_info["mlflow"] = {
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
        "logged_timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "model_uri": f"models:/{model_name}/{model_version}"
    }
    
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training info updated with MLflow details: {info_file}")

def log_model_to_mlflow(training_info):
    if not training_info:
        return False, None, None, None
    
    model_path = training_info["model_path"]
    metrics = training_info["metrics"]
    best_params = training_info["best_params"]
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False, None, None, None
    
    client = MlflowClient()
    
    model = get_or_create_model(client, FIXED_MODEL_NAME)
    if not model:
        return False, None, None, None
    
    expected_version = get_next_version(client, FIXED_MODEL_NAME)
    
    run_name = f"YOLO_v{expected_version}_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", "YOLO")
        mlflow.log_param("optimization", "Optuna")
        mlflow.log_param("version", expected_version)
        mlflow.log_param("training_timestamp", training_info["timestamp"])
        
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        mlflow.log_metric("mean_precision", metrics["mean_precision"])
        mlflow.log_metric("mAP_50", metrics["mAP50"])
        mlflow.log_metric("mean_recall", metrics["mean_recall"])
        mlflow.log_metric("mAP_50_90", metrics["mAP50_90"])
        
        mlflow.log_artifact(model_path, artifact_path="model")
        
        run_id = run.info.run_id
        print(f"Upload completed! Run ID: {run_id}")
    
    try:
        model_version = client.create_model_version(
            name=FIXED_MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=f"Optuna optimized version - mAP50_90: {metrics['mAP50_90']:.4f} | Architecture: {best_params['model']} | Optimizer: {best_params['optimizer']}"
        )
        
        print(f"Model version registered!")
        print(f"Name: {model_version.name}")
        print(f"Version: {model_version.version}")
        print(f"Performance: mAP50_90={metrics['mAP50_90']:.4f}")
        print(f"Architecture: {best_params['model']}")
        print(f"Optimizer: {best_params['optimizer']}")
        print(f"Access: http://localhost:5000")
        
        initial_tags = {
            "stage": "none",
            "architecture": best_params['model'],
            "optimizer": best_params['optimizer'],
            "created_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "performance_map50_90": f"{metrics['mAP50_90']:.4f}",
            "performance_map50": f"{metrics['mAP50']:.4f}",
            "training_method": "optuna_optimization"
        }
        
        for key, value in initial_tags.items():
            client.set_model_version_tag(
                name=FIXED_MODEL_NAME,
                version=model_version.version,
                key=key,
                value=str(value)
            )
        
        print("Initial tags added")
        
        return True, FIXED_MODEL_NAME, model_version.version, run_id
        
    except Exception as e:
        print(f"Registration error: {e}")
        return False, None, None, None

def show_model_history(client, model_name):
    """Show history of all versions for the model"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print("No versions found")
            return
        
        print(f"\nModel History for '{model_name}':")
        print("=" * 80)
        
        for version in sorted(versions, key=lambda x: int(x.version), reverse=True):
            stage = version.tags.get("stage", "none") if version.tags else "none"
            perf = version.tags.get("performance_map50_90", "N/A") if version.tags else "N/A"
            arch = version.tags.get("architecture", "N/A") if version.tags else "N/A"
            
            print(f"🔢 Version {version.version} | Stage: {stage} | mAP50_90: {perf} | Arch: {arch}")
            if version.description:
                print(f"   └── {version.description[:100]}...")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"Error showing model history: {e}")

def main():
    print("Starting MLflow logging process...")
    print(f"Using fixed model name: {FIXED_MODEL_NAME}")
    
    setup_mlflow_environment()
    client = MlflowClient()
    
    show_model_history(client, FIXED_MODEL_NAME)
    
    training_info = load_training_info()
    
    if training_info:
        print("\nTraining Results Summary:")
        print(f"   mAP50: {training_info['metrics']['mAP50']:.4f}")
        print(f"   mAP50_90: {training_info['metrics']['mAP50_90']:.4f}")
        print(f"   Mean Precision: {training_info['metrics']['mean_precision']:.4f}")
        print(f"   Mean Recall: {training_info['metrics']['mean_recall']:.4f}")
        print(f"   Best Model: {training_info['best_params']['model']}")
        print(f"   Best Optimizer: {training_info['best_params']['optimizer']}")
        print(f"   Training Time: {training_info['timestamp']}")
        
        success, model_name, model_version, run_id = log_model_to_mlflow(training_info)
        
        if success:
            update_training_info(training_info, model_name, model_version, run_id)
            
            print("\nModel logging completed successfully!")
            print(f"Model Name: {model_name}")
            print(f"Model Version: {model_version}")
            print("Next step: Run 04_promote_to_staging.py to promote to staging")
            
            show_model_history(client, model_name)
        else:
            print("\nModel logging failed!")
    else:
        print("\nNo training info found. Please run 02_train_model.py first.")

if __name__ == "__main__":
    main()