import functools
import gc

import torch

from mnemorai.logger import logger


def manage_memory(targets=None, delete_attrs=None, move_kwargs=None):
    """
    Decorator to manage memory by moving specified attributes to GPU before method call and back to CPU after method.

    Args:
        targets (list[str]): List of attribute names to move to GPU (e.g., ['model', 'pipe']).
        delete_attrs (list[str]): List of attribute names to delete after method execution.
        move_kwargs (dict): Additional keyword arguments to pass to the `.to()` method.

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
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Initialize the pipe if it's not already loaded
            if getattr(self, "pipe", None) is None:
                self._initialize_pipe()

            # Move specified targets to GPU if offloading is enabled
            if getattr(self, "offload", False):
                for target in targets:
                    attr = getattr(self, target, None)
                    if attr is not None:
                        logger.debug(f"Moving {target} to GPU (cuda).")
                        attr.to("cuda", **move_kwargs)

            try:
                # Execute the decorated method
                result = method(self, *args, **kwargs)
            finally:
                # Delete specified attributes if DELETE_AFTER_USE is True
                if self.config.get("DELETE_AFTER_USE", True):
                    for attr_name in delete_attrs:
                        attr = getattr(self, attr_name, None)
                        if attr is not None:
                            logger.debug(f"Deleting {attr_name} to free up memory.")
                            delattr(self, attr_name)
                            setattr(self, attr_name, None)
                    gc.collect()
                    torch.cuda.empty_cache()

                # Move specified targets back to CPU if offloading is enabled
                if getattr(self, "offload", False):
                    for target in targets:
                        attr = getattr(self, target, None)
                        if attr is not None:
                            logger.debug(f"Moving {target} back to CPU.")
                            attr.to("cpu", **move_kwargs)

            return result

        return wrapper

    return decorator
