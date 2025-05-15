from typing import Callable, Any, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map


def do_parallel(task_fn: Callable[[Any], Any], 
                arguments: List[Any], 
                n_workers: int, 
                use_tqdm: bool,
                mode: str = "process") -> List[Any]:
    assert isinstance(n_workers, int) and n_workers >= 0, f"n_workers must be int >=0, got {n_workers}"
    if n_workers == 0:
        result = []
        if use_tqdm:
            for arg in tqdm(arguments):
                result.append(task_fn(arg))
        else:
            for arg in arguments:
                result.append(task_fn(arg))
        return result
    
    else:
        if use_tqdm:
            if mode == "process":
                return process_map(task_fn, arguments, max_workers=n_workers)
            elif mode == "thread":
                return thread_map(task_fn, arguments, max_workers=n_workers)
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            if mode == "process":
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    result = executor.map(task_fn, arguments)
                    return [e for e in result]
            elif mode == "thread":
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    result = executor.map(task_fn, arguments)
                    return [e for e in result]
            else:
                raise ValueError(f"Invalid mode: {mode}")
