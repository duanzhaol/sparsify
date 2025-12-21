import os
from typing import Any, Type, TypeVar, cast

import torch
from accelerate.utils import send_to_device
from torch import Tensor, nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.base_model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
    hook_mode: str = "output",
) -> dict[str, int]:
    """Find number of input/output dimensions for the specified modules.

    Args:
        model: The model to inspect
        module_names: List of module names to check
        dim: Which dimension to extract (default: -1)
        hook_mode: "input" to measure input dimensions, "output" to measure output dimensions
    """
    module_to_name = {
        model.base_model.get_submodule(name): name for name in module_names
    }
    shapes: dict[str, int] = {}

    def hook(module, input, output):
        # Choose input or output based on hook_mode
        if hook_mode == "input":
            # Unpack input tuple (forward hooks receive input as tuple)
            if isinstance(input, tuple):
                tensor = input[0]
            else:
                tensor = input
        else:  # output mode
            # Unpack output tuples if needed
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

        name = module_to_name[module]
        shapes[name] = tensor.shape[dim]

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    dummy = send_to_device(model.dummy_inputs, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    return shapes


def get_max_layer_index(hookpoints: list[str], layers_name: str) -> int | None:
    """Extract maximum layer index from hookpoints.

    Args:
        hookpoints: List of hookpoint paths like "layers.0.self_attn.o_proj"
        layers_name: Name of the layers ModuleList (e.g., "layers" or "h")

    Returns:
        Maximum layer index if found, None otherwise
    """
    max_idx = -1
    base_name = layers_name.split('.')[0]  # Handle "layers" or "model.layers"

    for hookpoint in hookpoints:
        parts = hookpoint.split('.')
        # Check format: "layers.N.component" where N is integer
        if len(parts) >= 2 and parts[0] == base_name:
            try:
                layer_idx = int(parts[1])
                max_idx = max(max_idx, layer_idx)
            except ValueError:
                continue  # Skip non-numeric indices

    return max_idx if max_idx >= 0 else None


class StopForwardException(Exception):
    """Exception used to stop forward pass early."""
    pass


def partial_forward_to_layer(
    model: PreTrainedModel,
    input_ids: Tensor,
    max_layer_idx: int
) -> None:
    """Run forward pass only up to max_layer_idx to trigger necessary hooks.

    This avoids computing unnecessary layers when training SAEs on early layers.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        max_layer_idx: Maximum layer index to run (inclusive)
    """
    # Get the layer list
    _, layers = get_layer_list(model)

    # Register a hook on the last layer we need to stop after it
    stop_hook_handle = None

    def stop_hook(module, input):
        """Hook that raises exception to stop forward pass."""
        # Remove the hook immediately to avoid issues
        if stop_hook_handle is not None:
            stop_hook_handle.remove()
        raise StopForwardException()

    # Register hook on the layer AFTER the one we want (to stop after max_layer completes)
    if max_layer_idx < len(layers) - 1:
        stop_hook_handle = layers[max_layer_idx + 1].register_forward_pre_hook(stop_hook)

    try:
        # Run the full model forward - it will stop early due to our hook
        model(input_ids)
    except StopForwardException:
        # Expected - this means we successfully stopped early
        pass
    finally:
        # Clean up hook if it wasn't removed
        if stop_hook_handle is not None:
            stop_hook_handle.remove()


def set_submodule(model: nn.Module, submodule_path: str, new_submodule: nn.Module):
    """
    Replaces a submodule in a PyTorch model dynamically.

    Args:
        model (nn.Module): The root model containing the submodule.
        submodule_path (str): Dotted path to the submodule.
        new_submodule (nn.Module): The new module to replace the existing one.

    Example:
        set_submodule(model, "encoder.layer.0.attention.self", nn.Identity())
    """
    parent_path, _, last_name = submodule_path.rpartition(".")
    parent_module = model.get_submodule(parent_path) if parent_path else model
    setattr(parent_module, last_name, new_submodule)


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return nn.functional.embedding_bag(
        top_indices, W_dec.mT, per_sample_weights=top_acts, mode="sum"
    )


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return xformers_embedding_bag(top_indices, W_dec.mT, top_acts)


try:
    from .xformers import xformers_embedding_bag
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of sparse decoder.")
else:
    if os.environ.get("SPARSIFY_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of sparse decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string: str) -> dict:
    """
    Parses something like
        args1=val1,arg2=val2
    into a dictionary.
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        kv[0]: handle_arg_string("=".join(kv[1:]))
        for kv in [arg.split("=") for arg in arg_list]
    }
    return args_dict
