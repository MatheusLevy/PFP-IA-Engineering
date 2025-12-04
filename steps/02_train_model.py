import torch
import gc
import datetime as dt
import os
import optuna
from optuna.pruners import BasePruner
from optuna.trial import TrialState
from ultralytics import YOLO
import wandb
import json

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = [-1] * num_gpus
    return device

models = ["yolo12n", "yolo12s"]
data = "./dataset/data_freeze.yaml"
epochs = 10
patience = 100
min_batch_size = 2
max_batch_size = 5
batch_size_step = 2
image_size = [256, 320]
cache = True
optimizer = ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]
multi_scale = [True, False]
cos_lr = [True, False]
close_mosaic_min = 1
close_mosaic_max = 100
amp = True
lr0_min = 1e-5
lr0_max = 1e-1
dropout_min = 1e-5
dropout_max = 0.5
max_workers = max(1, (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) - 1)

def train(trial):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    group_name = f"study-{today}__build_{os.getenv('GIT_SHA','local')}__data_v42"
    run_name = f"trial/{trial.number}"
    
    run = wandb.init(
        project="yolo-optimization-monitor",
        entity="levybessa-puc",
        name=run_name,
        group=group_name,
        tags=["YOLO", "optuna"],
        reinit=True
    )
    
    wandb.log({
        "status": "running",
        "trial_number": trial.number,
        "stage": "initializing",
    })
    
    print(f"🚀 Started wandb run: {run_name} (ID: {run.id})")
    
    try:
        model_name = trial.suggest_categorical("model", models)
        batch_size = trial.suggest_int("batch_size", min_batch_size, max_batch_size, step=batch_size_step)
        imgsz = trial.suggest_categorical("imgsz", image_size)
        optimizer_name = trial.suggest_categorical("optimizer", optimizer)
        multi_scale_enabled = trial.suggest_categorical("multi_scale", multi_scale)
        lr0 = trial.suggest_float("lr0", lr0_min, lr0_max, log=True)
        dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
        
        hyperparams = {
            "trial_number": trial.number,
            "model": model_name,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "optimizer": optimizer_name,
            "multi_scale": multi_scale_enabled,
            "lr0": lr0,
            "dropout": dropout,
            "epochs": epochs,
            "patience": patience,
            "cache": cache,
            "amp": amp,
            "data": data,
            "max_workers": max_workers
        }
        
        wandb.config.update(hyperparams)
        
        print(f"📋 Trial {trial.number} parameters:")
        print(f"   Model: {model_name}, Batch: {batch_size}, ImgSz: {imgsz}")
        print(f"   Optimizer: {optimizer_name}, LR: {lr0:.6f}")
        print(f"   Group: {group_name}")
        print(f"   Run: {run_name}")
        print("-" * 70)
        
        model = YOLO(f"{model_name}.pt")
        
        print(f"🏋️ Starting training for trial {trial.number}...")
        
        results = model.train(
            project=None, 
            data=data,
            epochs=epochs,
            patience=patience,
            batch=batch_size,
            imgsz=imgsz,
            cache=cache,
            optimizer=optimizer_name,
            multi_scale=multi_scale_enabled,
            amp=amp,
            lr0=lr0,
            dropout=dropout,
            verbose=False,
            save=False,
            plots=True,
            device=get_device(),
            workers=max_workers
        )
        del model

        print("Limpando Memória...")
        torch.cuda.empty_cache()
        gc.collect()
        print("Memória Limpa.")

        mean_precision, mean_recall, mAP50, mAP50_90 = tuple(results.box.mean_results())
        
        final_metrics = {
            "mAP50_95": mAP50_90,
            "mean precision": mean_precision,
            "mean recall": mean_recall,
            "mAP50": mAP50,
            "status": "completed",
            "stage": "finished",
            "trial_number": trial.number,
        }
        
        wandb.log(final_metrics)
        wandb.finish()
        print(f"✅ Trial {trial.number} completed successfully: mAP = {mAP50_90:.4f}")
        print(f"🔗 Run URL: {run.url}")
        
        return mAP50
        
    except torch.cuda.OutOfMemoryError as e:
        error_info = {
            "status": "failed",
            "error_type": "CUDA_OOM", 
            "error_message": str(e),
            "trial_number": trial.number,
            "stage": "cuda_oom_error",
        }
        
        wandb.log(error_info)
        
        try:
            run.mark_preempting()
        except Exception as mark_error:
            print(f"⚠️ Could not mark as preempting: {mark_error}")
        
        print(f"💥 Trial {trial.number} - CUDA Out of Memory Error!")
        print(f"   Model: {model_name}, Batch: {batch_size}, ImgSz: {imgsz}")
        print(f"   Error: {str(e)}")
        print(f"   Device: {get_device()}")
        
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return None
        
    except Exception as e:
        error_info = {
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "trial_number": trial.number,
            "stage": "general_error",
        }
        
        wandb.log(error_info)
        
        try:
            wandb.finish(exit_code=1) 
        except Exception as mark_error:
            print(f"⚠️ Could not mark as preempting: {mark_error}")
        
        print(f"❌ Trial {trial.number} - Error: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return None
        
    finally:
        if 'model' in locals():
            try:
                del model
            except:
                pass
        
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
        
        print(f"🧹 Trial {trial.number} cleanup completed")

class ErrorPruner(BasePruner):
    def __init__(self, max_consecutive_errors=3):
        self.max_consecutive_errors = max_consecutive_errors
        self.error_count = 0

    def prune(self, study, trial):
        if trial.state == TrialState.FAIL:
            self.error_count += 1
            print(f"Trial {trial.number} failed. Consecutive errors: {self.error_count}")
            if self.error_count >= self.max_consecutive_errors:
                print(f"Many consecutive errors ({self.error_count}). Consider reviewing configuration.")
            return True
        else:
            self.error_count = 0
            return False

def optimize_hyperparameters(n_trials):
    study = optuna.create_study(direction="maximize",
                                study_name="YOLO_Hyperparameter_Optimization",
                                storage="sqlite:///yolo_hyperparameter_optimization.db",
                                load_if_exists=False,
                                pruner=ErrorPruner(max_consecutive_errors=5)
                                )
    study.optimize(train, n_trials=n_trials)
    
    print("Best hyperparameters found:")
    print("=" * 50)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    print(f"\nBest mAP50: {study.best_value:.4f}")
    
    return study

def save_training_info(best_params, final_results, output_file="training_info.json"):
    mean_precision, mean_recall, mAP50, mAP50_90 = tuple(final_results.box.mean_results())
    
    training_info = {
        "best_params": best_params,
        "metrics": {
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
            "mAP50": float(mAP50),
            "mAP50_90": float(mAP50_90)
        },
        "model_path": "./runs/detect/best_model/weights/best.pt",
        "timestamp": dt.datetime.utcnow().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"📁 Training info saved to: {output_file}")
    return training_info

if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    
    study = optimize_hyperparameters(n_trials=3)
    
    best_params = study.best_params
    final_model = YOLO(f"{best_params['model']}.pt")
    
    final_results = final_model.train(
        data=data,
        epochs=epochs,
        patience=patience,
        batch=best_params['batch_size'],
        imgsz=best_params['imgsz'],
        cache=cache,
        optimizer=best_params['optimizer'],
        multi_scale=best_params['multi_scale'],
        amp=amp,
        lr0=best_params['lr0'],
        dropout=best_params['dropout'],
        name='best_model',
        save=True,
        plots=True
    )
    
    save_training_info(best_params, final_results)
    
    print("Training completed!")
