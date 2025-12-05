import os
import mlflow
from mlflow import MlflowClient
import datetime as dt
import json
import tempfile
import glob
from ultralytics import YOLO
import torch

FIXED_MODEL_NAME = "yolo-rock-paper-scissors"

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


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = [-1] * num_gpus
    return device

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

def test_mlflow_connection(client):
    try:
        experiments = client.search_experiments()
        print(f"MLflow connection successful. Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return False

def find_model_by_stage(client, model_name, stage):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            if version.tags and version.tags.get("stage") == stage:
                print(f"Found {stage} model: {model_name} v{version.version}")
                return version
        
        print(f"No {stage} model found for: {model_name}")
        return None
        
    except Exception as e:
        print(f"Error searching for {stage} model: {e}")
        return None

def download_model_artifact(client, model_version):
    try:
        print(f"Downloading model {model_version.name} v{model_version.version}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_version.source,
                dst_path=temp_dir
            )
            
            pt_files = glob.glob(f"{temp_dir}/**/*.pt", recursive=True)
            
            if pt_files:
                model_path = pt_files[0]
                print(f"Model file found: {model_path}")
                
                model = YOLO(model_path)
                print(f"YOLO model v{model_version.version} loaded successfully!")
                return model, model_version
            else:
                print(".pt file not found in artifacts")
                return None, None
                
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None, None

def validate_model_with_dataset(model, model_version, data_path, description="Model"):
    try:
        print(f"\nValidating {description} v{model_version.version}...")
        
        results = model.val(
            data=data_path,
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.6,
            device=get_device(),
            plots=False,
            save_json=False,
            split="test",
            verbose=False
        )
        
        mean_precision, mean_recall, mAP50, mAP50_90 = tuple(results.box.mean_results())
        
        metrics = {
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
            "mAP50": float(mAP50),
            "mAP50_90": float(mAP50_90)
        }
        
        print(f"{description} Results:")
        print(f"   mAP50: {metrics['mAP50']:.4f}")
        print(f"   mAP50_90: {metrics['mAP50_90']:.4f}")
        print(f"   Precision: {metrics['mean_precision']:.4f}")
        print(f"   Recall: {metrics['mean_recall']:.4f}")
        
        return metrics, results
        
    except Exception as e:
        print(f"Error validating {description}: {e}")
        return None, None

def compare_models(staging_metrics, production_metrics=None, threshold=0.01):
    print(f"\nModel Comparison (threshold: {threshold:.3f}):")
    print("=" * 60)
    
    if production_metrics is None:
        print("No production model found - automatic promotion")
        return True, "no_production_model"
    
    staging_map50_90 = staging_metrics["mAP50_90"]
    production_map50_90 = production_metrics["mAP50_90"]
    
    staging_map50 = staging_metrics["mAP50"]
    production_map50 = production_metrics["mAP50"]
    
    print(f"Production Model:")
    print(f"   mAP50_90: {production_map50_90:.4f}")
    print(f"   mAP50: {production_map50:.4f}")
    
    print(f"Staging Model:")
    print(f"   mAP50_90: {staging_map50_90:.4f}")
    print(f"   mAP50: {staging_map50:.4f}")
    
    print(f"Differences:")
    print(f"   mAP50_90: {staging_map50_90 - production_map50_90:+.4f}")
    print(f"   mAP50: {staging_map50 - production_map50:+.4f}")
    
    map50_90_improved = staging_map50_90 >= (production_map50_90 + threshold)
    map50_improved = staging_map50 >= (production_map50 + threshold)
    
    print(f"\nValidation Results:")
    print(f"   mAP50_90 improvement >= {threshold:.3f}: {'PASS' if map50_90_improved else 'FAIL'}")
    print(f"   mAP50 improvement >= {threshold:.3f}: {'PASS' if map50_improved else 'FAIL'}")
    
    should_promote = map50_90_improved
    reason = "performance_improvement" if should_promote else "insufficient_improvement"
    
    if should_promote:
        print("Staging model qualifies for promotion!")
    else:
        print("Staging model does not qualify for promotion")
    
    return should_promote, reason

def promote_to_production(client, model_name, version, staging_metrics, reason):
    print(f"\nPromoting {model_name} v{version} to PRODUCTION...")
    
    try:
        demote_current_production(client, model_name, version)
        
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="stage",
            value="production"
        )
        print("Model tagged as PRODUCTION")
        
        performance_info = f"mAP50_90: {staging_metrics['mAP50_90']:.4f} | mAP50: {staging_metrics['mAP50']:.4f}"
        description = f"Production model - Validated and promoted | {performance_info}"
        
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print("Description updated")
        
        production_tags = {
            "validation_status": "passed",
            "performance_tier": "production",
            "promoted_to_production_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "promoted_by": "automated_validation",
            "promotion_reason": reason,
            "environment": "production",
            "validated_map50_90": f"{staging_metrics['mAP50_90']:.4f}",
            "validated_map50": f"{staging_metrics['mAP50']:.4f}"
        }
        
        for key, value in production_tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=str(value)
            )
        
        print("Production tags added successfully")
        return True
        
    except Exception as e:
        print(f"Error promoting to production: {e}")
        return False

def demote_current_production(client, model_name, new_version):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            if (version.tags and 
                version.tags.get("stage") == "production" and 
                version.version != new_version):
                
                print(f"Demoting current production model v{version.version} to archived...")
                
                client.set_model_version_tag(
                    name=model_name,
                    version=version.version,
                    key="stage",
                    value="archived"
                )
                
                client.set_model_version_tag(
                    name=model_name,
                    version=version.version,
                    key="archived_date",
                    value=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
                )
                
                print(f"Model v{version.version} demoted to archived")
                
    except Exception as e:
        print(f"Error demoting current production model: {e}")

def show_model_stages_summary(client, model_name):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        print(f"\nModel Stages Summary for '{model_name}':")
        print("=" * 80)
        
        stages = {"production": [], "staging": [], "archived": [], "none": []}
        
        for version in versions:
            stage = version.tags.get("stage", "none") if version.tags else "none"
            stages[stage].append(version)
        
        for stage, stage_versions in stages.items():
            if stage_versions:
                print(f"\n{stage.upper()}:")
                for version in sorted(stage_versions, key=lambda x: int(x.version)):
                    perf = version.tags.get("validated_map50_90", "N/A") if version.tags else "N/A"
                    print(f"   Version {version.version} | mAP50_90: {perf}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"Error showing model summary: {e}")

def main():
    print("Starting Production Validation Process...")
    print(f"Using model: {FIXED_MODEL_NAME}")
    
    setup_mlflow_environment()
    client = MlflowClient()
    
    if not test_mlflow_connection(client):
        print("Cannot connect to MLflow. Please check if MLflow server is running.")
        return
    
    training_info = load_training_info()
    if not training_info:
        return
    
    data_path = "./dataset/data_freeze.yaml"
    
    print("\n" + "="*70)
    print("PRODUCTION VALIDATION AND PROMOTION")
    print("="*70)
    
    show_model_stages_summary(client, FIXED_MODEL_NAME)
    
    staging_version = find_model_by_stage(client, FIXED_MODEL_NAME, "staging")
    if not staging_version:
        print("No staging model found. Please run 04_promote_to_staging.py first.")
        return
    
    production_version = find_model_by_stage(client, FIXED_MODEL_NAME, "production")
    
    staging_model, staging_version = download_model_artifact(client, staging_version)
    if not staging_model:
        print("Failed to download staging model")
        return
    
    staging_metrics, staging_results = validate_model_with_dataset(
        staging_model, staging_version, data_path, "Staging Model"
    )
    
    if not staging_metrics:
        print("Failed to validate staging model")
        return
    
    production_metrics = None
    if production_version:
        production_model, production_version = download_model_artifact(client, production_version)
        if production_model:
            production_metrics, production_results = validate_model_with_dataset(
                production_model, production_version, data_path, "Production Model"
            )
    
    should_promote, reason = compare_models(staging_metrics, production_metrics)
    
    if should_promote:
        print(f"\nPromotion Decision: PROMOTE")
        print(f"Reason: {reason}")
        
        success = promote_to_production(
            client, 
            FIXED_MODEL_NAME, 
            staging_version.version, 
            staging_metrics, 
            reason
        )
        
        if success:
            print(f"\nSUCCESS! Model v{staging_version.version} promoted to PRODUCTION!")
            print("Access MLflow UI: http://localhost:5000")
            
            show_model_stages_summary(client, FIXED_MODEL_NAME)
        else:
            print("\nPromotion failed!")
    else:
        print(f"\nPromotion Decision: DO NOT PROMOTE")
        print(f"Reason: {reason}")
        print("Model remains in staging. Improve performance and try again.")

if __name__ == "__main__":
    main()