import functools
import gc
import inspect

import torch

from mnemorai.logger import logger


def manage_memory(targets=None, delete_attrs=None, move_kwargs=None):
    """
    Decorator to manage memory by moving specified attributes to GPU before method call and back to CPU after method.

    Args:
        targets (list[str]): List of attribute names to move to GPU (e.g., ['model', 'pipe']).
        delete_attrs (list[str]): List of attribute names to delete after method execution.
        move_kwargs (dict): Additional keyword arguments to pass to the .to() method.

    Returns
    -------
    function: Decorated method.
    """
    if targets is None:
        targets = []
    if delete_attrs is None:
        delete_attrs = []
    if move_kwargs is None:
        move_kwargs = {}

    def decorator(method):
        is_async = inspect.iscoroutinefunction(method)

        async def _async_wrapper(self, *args, **kwargs):
            # before
            if getattr(self, "pipe", None) is None:
                self._initialize_pipe()
            if getattr(self, "offload", False):
                for t in targets:
                    obj = getattr(self, t, None)
                    if obj is not None:
                        logger.debug(f"Moving {t} to GPU.")
                        obj.to("cuda", **move_kwargs)

            try:
                result = await method(self, *args, **kwargs)
            finally:
                # delete attrs
                if self.config.get("DELETE_AFTER_USE", True):
                    for name in delete_attrs:
                        if hasattr(self, name):
                            logger.debug(f"Deleting {name} to free memory.")
                            delattr(self, name)
                            setattr(self, name, None)
                    gc.collect()
                    torch.cuda.empty_cache()

                # move back
                if getattr(self, "offload", False):
                    for t in targets:
                        obj = getattr(self, t, None)
                        if obj is not None:
                            logger.debug(f"Moving {t} back to CPU.")
                            obj.to("cpu", **move_kwargs)

            return result

        def _sync_wrapper(self, *args, **kwargs):
            # identical pre/post logic, but sync
            if getattr(self, "pipe", None) is None:
                self._initialize_pipe()
            if getattr(self, "offload", False):
                for t in targets:
                    obj = getattr(self, t, None)
                    if obj is not None:
                        logger.debug(f"Moving {t} to GPU.")
                        obj.to("cuda", **move_kwargs)

            try:
                result = method(self, *args, **kwargs)
            finally:
                if self.config.get("DELETE_AFTER_USE", True):
                    for name in delete_attrs:
                        if hasattr(self, name):
                            logger.debug(f"Deleting {name} to free memory.")
                            delattr(self, name)
                            setattr(self, name, None)
                    gc.collect()
                    torch.cuda.empty_cache()

                if getattr(self, "offload", False):
                    for t in targets:
                        obj = getattr(self, t, None)
                        if obj is not None:
                            logger.debug(f"Moving {t} back to CPU.")
                            obj.to("cpu", **move_kwargs)

            return result

        wrapper = _async_wrapper if is_async else _sync_wrapper
        return functools.wraps(method)(wrapper)

    return decorator
