"""
Thin wrapper around MLflow experiment tracking.

Reads MLFLOW_TRACKING_URI from the environment (MLflow also picks this up
natively). If set, returns a live tracker backed by mlflow. If unset or
mlflow is unavailable, returns a no-op stub — callers never need to branch.

Usage:
    run = make_mlflow_run("bank-marketing", config="roll-aoc", episode=0)
    run["hparams"] = {"lr": 0.001, "optim": "Adam"}
    run.track(0.42, name="loss", step=5, context={"split": "train"})
    run.close()

Set MLFLOW_TRACKING_URI=http://192.168.1.100:5000 in the environment to enable.
"""
import os
import logging


class _NoOpRun:
    def track(self, *args, **kwargs): pass
    def __setitem__(self, key, value): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _MLFlowRun:
    def track(self, value, name, step, context=None):
        import mlflow
        prefix = (context.get('split', '') + '/') if context else ''
        try:
            mlflow.log_metric(prefix + name, float(value), step=step)
        except Exception as exc:
            logging.warning(f'MLflow log_metric failed (suppressed): {exc}')

    def __setitem__(self, key, value):
        import mlflow
        if isinstance(value, dict):
            mlflow.log_params({f"{key}.{k}": str(v) for k, v in value.items()})
        else:
            mlflow.set_tag(key, str(value))

    def close(self):
        import mlflow
        mlflow.end_run()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def make_mlflow_run(experiment_name: str, **tags) -> _NoOpRun:
    """Return an _MLFlowRun if MLFLOW_TRACKING_URI is set, else a no-op stub."""
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        return _NoOpRun()
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(tags={k: str(v) for k, v in tags.items()})
        return _MLFlowRun()
    except Exception as exc:
        logging.warning(f'MLflow tracking unavailable ({exc}) — proceeding without it')
        return _NoOpRun()
