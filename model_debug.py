import sys
import inspect
from typing import Tuple, List
import argparse
import pathlib
from loguru import logger
import torch, torch.nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin

LOG_LEVEL_SUMMARY = "SUMMARY"
LOG_LEVEL_INPUT = "INPUT"
LOG_LEVEL_OUTPUT = "OUTPUT"
LOG_LEVEL_PARAMS = "PARAMS"
LOG_LEVEL_HIGHLIGHT = "HIGHLIGHT"

# Colors: https://htmlcolorcodes.com/color-names/
# NOTE: Existing Loguru log levels: INFO=25, WARNING=30
logger.level(LOG_LEVEL_SUMMARY, no=26, color="<fg #FFD700>")   # Gold
logger.level(LOG_LEVEL_INPUT, no=27, color="<fg #87CEFA>")     # LightSkyBlue
logger.level(LOG_LEVEL_OUTPUT, no=28, color="<fg #00BFFF>")    # DeepSkyBlue
logger.level(LOG_LEVEL_PARAMS, no=28, color="<fg #4682B4>")    # SteelBlue
logger.level(LOG_LEVEL_HIGHLIGHT, no=31, color="<fg #00FF00>") # Green
logger.remove()
logger.add(sys.stderr, level="INFO")

DEFAULT_USER_PROMPT = "What color is the sky?"

#tree_branch = '\u2514' # ╚
# def print_mro_tree(cls, indent=0, branch_char=""):
#     logger.log(LOG_LEVEL_SUMMARY, f"{' ' * indent} + inset + {cls.__name__}")
#     for base in cls.__bases__:
#         print_mro_tree(base, indent + 1, branch_char=tree_branch)

# import keyboard
# def pause_for_key():
#     while True:
#         event = keyboard.read_event(suppress=True)
#         if event.event_type == keyboard.KEY_DOWN:
#             return event.name

class DebugFlags:
    def __init__(self, debug:bool, trace:bool, pause:bool):
        self.pause = pause
        self.debug = debug
        self.trace = trace
    def __str__(self):
        return f"DebugFlags(pause={self.pause}, debug={self.debug}, trace={self.trace})"

def pause_until_key_pressed():
    user_input = input(f"Press <Enter> to continue...")
    if user_input != "":
        logger.trace(f"User pressed '{user_input}'")

def getShape(t) -> Tuple[bool, str]:
    return (True, str(t.shape)) if isinstance(t, torch.Tensor) else (False, "NONE")

# Note: all parameters are tensors (i.e., have tensor "data")
# see: https://pytorch.org/docs/stable/tensors.html#torch.Tensor
def format_module_parameters(filter:List[str]=None, indent=2) -> str:
    info = ""
    for name, param in module.named_parameters():
        if not filter or (name in filter):
            info += f"\n{' ' * indent}name: '{name}': shape: {param.shape},\n{' ' * indent}value: {param.data}"
    return info

def print_module_parameters(filter:List[str]=None, indent=2) -> str:
    info = format_module_parameters(filter=filter, indent=indent)
    if info:
        logger.log(LOG_LEVEL_PARAMS, f"\n[{LOG_LEVEL_PARAMS}]:{info}")
    return info

def print_torch_tensors(hook_name:str, level:str, module_name:str, module_class_name:str, data, indent:int=2):
    shape = "UNKNOWN"
    msg = "\n"
    if len(data) > 0:
        for i, tensor in enumerate(data):
            if type(tensor) is torch.Tensor:
                if i==0:
                    hasShape, shape = getShape(tensor)
                    if not hasShape:
                        logger.warning(f"[{level}] Tensor '{tensor}' has no attribute 'shape'.")
                    msg += f"[{level}] {module_name}:{module_class_name} ({hook_name}): {shape}:"
                msg += f"\n[{i}]:{tensor}"
                logger.log(level, f"{msg}")
            else:
                logger.warning(f"Unknown module data type: {type(tensor)} expected 'torch.Tensor'")
    else:
        logger.warning(f"\n[{level}] {module_name} ({hook_name}): No tensor data found.")
    return

# Generate our hook function (lambda) and use param. capture to save layer-specific info.
def create_forward_pre_hook_with_name(module_name, flags:DebugFlags):
    # A pre-forward hook is attached to a specific layer, and its callback function
    # is triggered BEFORE the layer's forward() method is executed.
    def forward_pre_hook(module, input):
        module_class_name = module.__class__.__name__
        print_torch_tensors("pre_forward", LOG_LEVEL_INPUT, module_name, module_class_name, input)
        if flags.pause:
            pause_until_key_pressed()
        return input

    return forward_pre_hook

# Generate our hook function (lambda) and use param. capture to save layer-specific info.
def create_forward_hook_with_name(module_name, flags:DebugFlags):
    # A forward hook is attached to a specific layer and its callback function
    # is triggered immediately AFTER the layer's forward() method is executed.
    def forward_hook(module, input, output): # , module_name
        module_class_name = module.__class__.__name__
        logger.log(LOG_LEVEL_HIGHLIGHT, f"{module_name}: {str(module)}")
        logger.debug(f"flags={flags}")
        print_module_parameters(filter=["weight"])
        print_torch_tensors(f"forward", LOG_LEVEL_INPUT, module_name, module_class_name, input)
        print_torch_tensors(f"forward", LOG_LEVEL_OUTPUT, module_name, module_class_name, output)
        if flags.pause:
            pause_until_key_pressed()

    return forward_hook

def filter_match(module_name:str, class_name:str, filter_class_name, filter_model_name) -> bool:
    # If no filter provided, no modules are a "match"
    if not filter_model_name and not filter_class_name:
        return True

    if module_name:
        for fname in filter_model_name:
            if fname in module_name:
                return True
    return False

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        parser.add_argument('-m', '--model-path', type=pathlib.Path, required=True, help='path to local HF model repo.')
        parser.add_argument('-p', '--prompt', required=False, default=DEFAULT_USER_PROMPT, help='Optional prompt text on model.generate()')
        parser.add_argument('--hook-pre-forward', default=False, action='store_true', help='Enable pre-forward hook on modules.')
        parser.add_argument('--class-hierarchy', default=True, action='store_true', help='Show module class hierarchy')
        parser.add_argument('--filter-class', type=str, nargs='+', required=False, help='Include only modules whose class name includes the substrings provided.')
        parser.add_argument('--filter-name', type=str, nargs='+', required=False, help='Include only modules whose name includes the substrings provided.')
        parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose output')
        parser.add_argument('--debug', default=False, action='store_true', help='Enable debug output')
        parser.add_argument('--trace', default=False, action='store_true', help='Enable trace output')
        parser.add_argument('--pause', default=False, action='store_true', help='Pause between forward hooks')
        parser.add_argument('--truncate-threshold', default=10, type=int, help='truncate tensor value printout if > threshold')
        args = parser.parse_args()

        if args.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        if args.debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        # store flags for use in hook callbacks
        flags:DebugFlags = DebugFlags(args.debug, args.trace, args.pause)
        logger.trace(f"flags: {flags}")

        local_path = str(args.model_path)
        if not args.model_path.exists():
            raise ValueError(f"--model-path {local_path} does not exist")

        if not args.model_path.is_dir():
            raise ValueError(f"--model-path {local_path} is not a model repo. (directory)")

        # TODO: allow cmd. line config of print opts.
        threshold = args.truncate_threshold
        torch.set_printoptions(threshold=threshold, linewidth=200, edgeitems=3) # others: precision=4, sci_mode=False

        # load the model
        model = AutoModelForCausalLM.from_pretrained(local_path)

        # Register "hooks" matching the following filters:
        if args.filter_class:
            logger.info(f"Registering hooks: for module_class: {args.filter_class}")

        if args.filter_name:
            logger.info(f"Registering hooks for module_names: {args.filter_name}")

        # Note: the only means to obtain the module's "tensor" name (not class name)
        # is using the following method:
        named_modules = model.named_modules()  # return (str, Module) – Tuple of module name and module
        for idx, module_tuple in enumerate(named_modules):
            module_name = module_tuple[0]
            module = module_tuple[1]
            class_name = module.__class__.__name__
            module_type = type(module)

            # Get some interesting features of the module
            isTorchModule = issubclass(module.__class__, torch.nn.Module)
            isPretrainedModel = issubclass(module.__class__, PreTrainedModel)
            hasUtils = issubclass(module.__class__, ModuleUtilsMixin)

            # Display trace info for every module
            logger.trace(f"[{idx:02d}]:({module_name}): classname: `{class_name}`, isPretrainedModel: {isPretrainedModel}, hasModuleUtilsMixin: {hasUtils}")

            if not isTorchModule:
                logger.warning(f"Skipping non-torch module[{idx}] type: ({module_type}), classname: `{class_name}`")
                continue

            # Only print the class hierarchy (once) using the top-level model, if requested
            if idx == 0 and args.class_hierarchy:
                logger.log(LOG_LEVEL_SUMMARY, f"Module class hierarchy:\n{module}")

            # Register requested hooks for each module
            if filter_match(module_name, class_name, args.filter_class, args.filter_name):
                if args.hook_pre_forward:
                    logger.info(f">> registering forward pre-hook: {module_name}...")
                    module.register_forward_pre_hook(create_forward_pre_hook_with_name(module_name, flags))
                logger.info(f">> registering forward hook: {module_name}:{str(module)}...")
                module.register_forward_hook(create_forward_hook_with_name(module_name, flags))

        # tokenize the user input prompt
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        prompt = args.prompt
        logger.info(f"tokenizer(): \"{prompt}\"")
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            logger.info(f"calling model.generate():\n  inputs.input_ids: {inputs.input_ids}")
            outputs = model.generate(**inputs)
        logger.info(f"calling tokenizer.decode():\n  outputs[0] ({len(outputs)}): {outputs[0]}")
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f">> {decoded_output}")

    except SystemExit as se:
        print(f"Usage: {parser.format_usage()}")
        exit(se)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Usage: {parser.format_usage()}")
        exit(2)

    # Exit successfully
    sys.exit(0)
