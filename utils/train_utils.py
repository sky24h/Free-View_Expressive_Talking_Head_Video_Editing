import time
import os
import glob
from natsort import natsorted


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            # raise TimerError(f"Timer is running. Use .stop() to stop it")
            self._start_time = None

        self._start_time = time.perf_counter()

    def stop(self, category=None):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        # print(f"Elapsed time for {category}: {elapsed_time:0.4f} seconds")
        return elapsed_time


def get_first_and_interval(checkpoint_names):
    splits = checkpoint_names[0].split("_")
    for s in splits:
        if "step" in s:
            first_step = int(s.replace("step", "").replace(".pth", ""))
            break

    splits = checkpoint_names[1].split("_")
    for s in splits:
        if "step" in s:
            second_step = int(s.replace("step", "").replace(".pth", ""))
            break
    return first_step, second_step - first_step


def check_saved_checkpoints(checkpoint_dir, prefix):
    keep_best_num = 5
    if prefix == "gen":
        size_limit = 100
    elif prefix == "sync":
        size_limit = 40
    else:
        raise ValueError("prefix should be 'gen' or 'sync'")
    # get current checkpoints
    saved_checkpoints = natsorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}_checkpoint*")))
    # only keep the latest 5 best models
    best_models = natsorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}_best_model_*")))
    if len(best_models) > keep_best_num:
        best_models = best_models[: len(best_models) - keep_best_num]
        for f in best_models:
            print("Remove old best model:", f)
            os.remove(f)
    # only keep {size_limit}G of all checkpoints
    total_size = sum([os.path.getsize(f) for f in saved_checkpoints]) / 1024 / 1024 / 1024
    print(f"Total size of checkpoints: {total_size}")
    if total_size > size_limit:
        # find best interval
        change_intervals = int(total_size / size_limit)
        first_step, current_interval = get_first_and_interval(saved_checkpoints[:2])
        target_interval = current_interval * (2**change_intervals)
        print("change interval from {} to {}".format(current_interval, target_interval))
        need_to_be_removed = natsorted(list(set(saved_checkpoints) - set(saved_checkpoints[::target_interval])))
        for f in need_to_be_removed:
            print("Remove checkpoint:", f)
            os.remove(os.path.join(checkpoint_dir, f))
        saved_checkpoints = natsorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}_checkpoint*")))
        size_after_remove = sum([os.path.getsize(f) for f in saved_checkpoints]) / 1024 / 1024 / 1024
        print(f"Total size of checkpoints after remove: {size_after_remove}")
        return target_interval
    else:
        print("No need to remove")
