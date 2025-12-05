import os
import mlflow
from mlflow import MlflowClient
import datetime as dt
import json

def setup_mlflow_environment():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI não está definida no ambiente")
    if not s3_endpoint:
        raise RuntimeError("MLFLOW_S3_ENDPOINT_URL não está definida no ambiente")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"S3 Endpoint URL: {s3_endpoint}")

def load_training_info(info_file="training_info.json"):
    try:
        with open(info_file, 'r') as f:
            training_info = json.load(f)
        print(f"Training info loaded from: {info_file}")
        return training_info
    except FileNotFoundError:
        print(f"Training info file not found: {info_file}")
        print("Please run 02_train_model.py and 03_log_model.py first")
        return None
    except Exception as e:
        print(f"Error loading training info: {e}")
        return None

def validate_training_info(training_info):
    """Validate that training_info has MLflow details"""
    if not training_info:
        return False, "No training info found"
    
    if "mlflow" not in training_info:
        return False, "No MLflow information found. Please run 03_log_model.py first"
    
    mlflow_info = training_info["mlflow"]
    required_fields = ["model_name", "model_version", "run_id"]
    
    for field in required_fields:
        if field not in mlflow_info:
            return False, f"Missing MLflow field: {field}"
    
    return True, "Valid"

def test_mlflow_connection(client):
    try:
        experiments = client.search_experiments()
        print(f"MLflow connection successful. Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return False

def validate_model_exists(client, model_name, version):
    try:
        model_version = client.get_model_version(name=model_name, version=version)
        print(f"Model {model_name} v{version} found")
        tags = model_version.tags
        current_stage = tags.get("stage", "None")
        print(f"   Current stage (from tag): {current_stage}")
        print(f"   Created: {model_version.creation_timestamp}")
        print(f"   Source: {model_version.source}")
        return True, model_version
    except Exception as e:
        print(f"Model {model_name} v{version} not found: {e}")
        return False, None

def validate_model_performance(training_info):
    if not training_info:
        return False
    
    metrics = training_info.get("metrics", {})
    mAP50_90 = metrics.get("mAP50_90", 0)
    mAP50 = metrics.get("mAP50", 0)
    mean_precision = metrics.get("mean_precision", 0)
    mean_recall = metrics.get("mean_recall", 0)
    
    print("\nPerformance Validation:")
    print(f"   mAP50_90: {mAP50_90:.4f}")
    print(f"   mAP50: {mAP50:.4f}")
    print(f"   Precision: {mean_precision:.4f}")
    print(f"   Recall: {mean_recall:.4f}")
    
    min_map50_90 = 0.5
    min_map50 = 0.6
    min_precision = 0.7
    min_recall = 0.6
    
    validation_passed = (
        mAP50_90 >= min_map50_90 and
        mAP50 >= min_map50 and
        mean_precision >= min_precision and
        mean_recall >= min_recall
    )
    
    print("\nValidation Requirements:")
    print(f"   mAP50_90 >= {min_map50_90}: {'PASS' if mAP50_90 >= min_map50_90 else 'FAIL'}")
    print(f"   mAP50 >= {min_map50}: {'PASS' if mAP50 >= min_map50 else 'FAIL'}")
    print(f"   Precision >= {min_precision}: {'PASS' if mean_precision >= min_precision else 'FAIL'}")
    print(f"   Recall >= {min_recall}: {'PASS' if mean_recall >= min_recall else 'FAIL'}")
    
    if validation_passed:
        print("Performance validation PASSED")
    else:
        print("Performance validation FAILED")
    
    return validation_passed

def promote_to_staging(client, model_name, version, training_info):
    print(f"\nTagging {model_name} v{version} as STAGING...")
    
    try:
        current_version = client.get_model_version(name=model_name, version=version)
        tags = current_version.tags
        
        current_stage = tags.get("stage", None)
        
        if current_stage == "staging":
            print("Model is already tagged as Staging")
            return True
            
        if current_stage == "production":
            print("Model is currently tagged as Production")
            print("Tag update cancelled: non-interactive mode")
            return False
    except Exception as e:
        print(f"Error getting current model state: {e}")
        return False
    
    try:
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="stage",
            value="staging"
        )
        print("Model tagged as STAGING")
    except Exception as e:
        print(f"Error setting stage tag: {e}")
        return False
    
    try:
        metrics = training_info.get("metrics", {})
        performance_info = f"mAP50_90: {metrics.get('mAP50_90', 0):.4f} | mAP50: {metrics.get('mAP50', 0):.4f}"
        description = f"Model in staging - Performance validated | {performance_info}"
        
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print("Description updated")
    except Exception as e:
        print(f"Error updating description: {e}")
    
    try:
        validation_status = "passed" if validate_model_performance(training_info) else "pending"
        
        tags = {
            "validation_status": validation_status,
            "performance_tier": "high" if validation_status == "passed" else "medium",
            "promoted_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "promoted_by": "automated_pipeline",
            "environment": "staging"  # Tag adicional para ambiente
        }
        
        if "best_params" in training_info:
            tags["model_architecture"] = training_info["best_params"].get("model", "unknown")
            tags["optimizer"] = training_info["best_params"].get("optimizer", "unknown")
        
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=str(value)
            )
        
        print("Tags added successfully")
    except Exception as e:
        print(f"Error adding tags: {e}")
    return True

def check_model_stage(client, model_name, version):
    """Verificar o estágio atual baseado nas tags"""
    try:
        model_version = client.get_model_version(name=model_name, version=version)
        tags = model_version.tags
        stage = tags.get("stage", "None")
        return stage
    except Exception as e:
        print(f"Error checking model stage: {e}")
        return None

def main():
    print("Starting model tagging for staging...")
    
    setup_mlflow_environment()
    client = MlflowClient()
    
    if not test_mlflow_connection(client):
        print("Cannot connect to MLflow. Please check if MLflow server is running.")
        return
    
    training_info = load_training_info()
    
    is_valid, message = validate_training_info(training_info)
    if not is_valid:
        print(f"{message}")
        return
    
    mlflow_info = training_info["mlflow"]
    model_name = mlflow_info["model_name"]
    model_version = mlflow_info["model_version"]
    
    print("\n" + "="*60)
    print("MODEL TAGGING FOR STAGING")
    print(f"Model: {model_name}")
    print(f"Version: {model_version}")
    print(f"Logged: {mlflow_info.get('logged_timestamp', 'Unknown')}")
    
    exists, model_version_obj = validate_model_exists(client, model_name, model_version)
    if not exists:
        print("Model validation failed. Exiting...")
        return
    
    if training_info.get("metrics"):
        print("\nModel Performance:")
        metrics = training_info["metrics"]
        print(f"   mAP50: {metrics.get('mAP50', 0):.4f}")
        print(f"   Precision: {metrics.get('mean_precision', 0):.4f}")
        print(f"   Recall: {metrics.get('mean_recall', 0):.4f}")
    
    validation_passed = validate_model_performance(training_info)
    
    if not validation_passed:
        print("\nWARNING: Model didn't pass all performance validations")
        print("Tagging cancelled due to failing validations (non-interactive mode)")
        return
    
    success = promote_to_staging(client, model_name, model_version, training_info)
    
    if success:
        print(f"\nModel {model_name} v{model_version} successfully tagged as STAGING!")
        current_stage = check_model_stage(client, model_name, model_version)
        print(f"Current stage tag: {current_stage}")
    else:
        print(f"\nFailed to tag model {model_name} v{model_version}")

if __name__ == "__main__":
    main()