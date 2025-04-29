from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import sys
import inspect
import torch
import argparse
import pathlib
from loguru import logger

# See: https://htmlcolorcodes.com/color-names/
# Gold:	#FFD700
# Bright green: #00FF00
# DarkViolet: #9400D3
# Light cyan: #E0FFFF
# Pale turquoise: #AFEEEE
# Aquamarine: #7FFFD4
# LightSkyBlue: #87CEFA
# DeepSkyBlue: #00BFFF
# DodgerBlue: #1E90FF
logger.level("SUMMARY", no=26, color="<fg #FFD700>")  # INFO=25, WARNING=30
logger.level("INPUT", no=27, color="<fg #87CEFA>")
logger.level("OUTPUT", no=28, color="<fg #00BFFF>")
logger.level("HIGHLIGHT", no=29, color="<fg #00FF00>")
logger.remove()
logger.add(sys.stderr, level="INFO")

import torch, torch.nn

# TODO: allow cmd. line config of print opts.
torch.set_printoptions(threshold=1000, linewidth=200, edgeitems=3) # edgeitems=3, precision=4, sci_mode=False

def print_torch_tensor(level:str, module_name:str, data, indent:int=2):
    for i, t in enumerate(data):
        if i==0:
            shape = t.shape
            msg = f"\n>>> {module_name}, {shape}:"
        msg += f"\n[{i}]:{t}"
    logger.log(level, f"{msg}")
    return

# A pre-forward hook is attached to a specific layer, and its callback function
# is triggered BEFORE the forward() method of the layer is executed.
# Enables:
# - Inspection of the input data and identify potential issues in the data or mode
def forward_pre_hook_function(module, input):
    module_name = module.__class__.__name__
    for i, tensor_input in enumerate(input):
        logger.log("INPUT", f"\n>>> {module_name}, tensor:\n INPUT[{i}]: {tensor_input}")
    return input

# A forward hook is attached to a specific layer and its callback function
# is triggered immediately AFTER the layer's forward() method is executed.
# Enables:
# - Visualizing the results of layer activations to gain insights into the module's behavior
def forward_hook_function(module, input, output):
    module_name = module.__class__.__name__
    print_torch_tensor("INPUT", module_name, input)
    print_torch_tensor("OUTPUT", module_name, output)
    return output

def filter_match(module_name:str, module: torch.nn.Module, filter_class_name, filter_model_name) -> bool:
    class_name = module.__class__.__name__
    if module_name:
        for fname in filter_model_name:
            if fname in module_name:
                return True
    return False

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        parser.add_argument('-m', '--model-path', type=pathlib.Path, required=True, help='path to local HF model repo.')
        # TODO: default to false?
        parser.add_argument('--pre-forward-hook', default=False, action='store_true', help='Enable pre-forward hook on modules.')
        parser.add_argument('--class-hierarchy', default=True, action='store_true', help='Show module class hierarchy')
        parser.add_argument('--module-class', type=str, nargs='+', required=False, help='Include only modules whose class name includes the substring provided.')
        parser.add_argument('--module-name', type=str, nargs='+', required=False, help='Include only modules whose name includes the substring provided.')
        parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose output')
        parser.add_argument('--debug', default=False, action='store_false', help='Enable debug output')
        args = parser.parse_args()

        if args.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        if args.debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        local_path = str(args.model_path)
        if not args.model_path.exists():
            raise ValueError(f"--model-path {local_path} does not exist")

        if not args.model_path.is_dir():
            raise ValueError(f"--model-path {local_path} is not a model repo. (directory)")

        if args.module_class:
            logger.info(f"filtering modules with class name: {args.module_class}")

        if args.module_name:
            logger.info(f"filtering modules with names containing: {args.module_name}")

        model = AutoModelForCausalLM.from_pretrained(local_path)

        # return (str, Module) – Tuple of name and module
        named_modules = model.named_modules()
        for idx, module_tuple in enumerate(named_modules):
            module_name = module_tuple[0]
            module = module_tuple[1]
            class_name = module.__class__.__name__
            module_type = type(module)
            logger.debug(f"[{idx:02d}]:({module_name}): {class_name}")

            isTorchModule = issubclass(module.__class__, torch.nn.Module)
            isPretrainedModel = issubclass(module.__class__, PreTrainedModel)
            logger.trace(f"module[{idx}] type: ({module_type}), classname: `{class_name}`, isPretrainedModel: {isPretrainedModel}")

            if not isTorchModule:
                logger.warning(f"Skipping non-torch module[{idx}] type: ({module_type}), classname: `{class_name}`")
                continue

            if idx == 0 and args.class_hierarchy:
                logger.log("SUMMARY", f"Module class hierarchy:\n`{module}`")

            # Register hooks for each module
            # logger.debug(f"testing module_name: {module_name} against filter: {args.module_name}")
            if filter_match(module_name, module, args.module_class, args.module_name):
                if args.pre_forward_hook:
                    logger.info(f">> registering forward pre-hook for module_name: {module_name}...")
                    module.register_forward_pre_hook(forward_pre_hook_function)
                logger.info(f">> registering forward hook for module_name: {module_name}...")
                module.register_forward_hook(forward_hook_function)

        tokenizer = AutoTokenizer.from_pretrained(local_path)
        inputs = tokenizer("Hello World", return_tensors="pt")
        print(inputs.input_ids)

        with torch.no_grad():
            outputs = model.generate(**inputs)
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f">> {gen}")

    except SystemExit as se:
        print(f"Usage: {parser.format_usage()}")
        exit(se)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Usage: {parser.format_usage()}")
        exit(2)

    # Exit successfully
    sys.exit(0)
