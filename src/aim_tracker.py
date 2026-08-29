"""
Thin wrapper around Aim experiment tracking.

Reads AIM_SERVER_URI from the environment. If set, returns a real aim.Run
pointed at that server. If unset or aim is unavailable, returns a no-op stub
with the same interface — callers never need to branch.

Usage:
    run = make_aim_run(experiment_name="bank-marketing", config="roll-aoc", episode=0)
    run["hparams"] = {"lr": 0.001}
    run.track(0.42, name="loss", step=5, context={"split": "train"})
    run.close()

Set AIM_SERVER_URI=aim://192.168.1.100:53800 in the environment to enable.
"""
import os
import logging


class _NoOpRun:
    """Drop-in for aim.Run when tracking is disabled."""
    def track(self, *args, **kwargs):
        pass
    def __setitem__(self, key, value):
        pass
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def make_aim_run(experiment_name: str, **tags) -> _NoOpRun:
    """Return an aim.Run if AIM_SERVER_URI is set, else a no-op stub.

    experiment_name: groups runs in the Aim UI (use dataset name).
    tags: arbitrary key-value pairs stored on the run (config, episode, device…).
    """
    uri = os.environ.get('AIM_SERVER_URI')
    if not uri:
        return _NoOpRun()
    try:
        import aim
        run = aim.Run(repo=uri, experiment=experiment_name)
        for k, v in tags.items():
            run[k] = v
        return run
    except Exception as exc:
        logging.warning(f'Aim tracking unavailable ({exc}) — proceeding without it')
        return _NoOpRun()
