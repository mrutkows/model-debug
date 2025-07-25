import sys
from typing import Tuple, List
import argparse
import pathlib
from loguru import logger
import torch, torch.nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel, AutoModelForVision2Seq
from transformers.modeling_utils import ModuleUtilsMixin

LOG_LEVEL_SUMMARY = "SUMMARY"
LOG_LEVEL_INPUT = "INPUT"
LOG_LEVEL_OUTPUT = "OUTPUT"
LOG_LEVEL_PARAMS = "PARAMS"
LOG_LEVEL_HIGHLIGHT = "HIGHLIGHT"
LOG_LEVEL_ALERT = "ALERT"

# Colors: https://htmlcolorcodes.com/
# NOTE: Existing Loguru log levels: INFO=25, WARNING=30
logger.level(LOG_LEVEL_SUMMARY, no=26, color="<fg #FFD700>")   # Gold
logger.level(LOG_LEVEL_INPUT, no=27, color="<fg #87CEFA>")     # LightSkyBlue
logger.level(LOG_LEVEL_OUTPUT, no=28, color="<fg #00BFFF>")    # DeepSkyBlue
logger.level(LOG_LEVEL_PARAMS, no=28, color="<fg #4682B4>")    # SteelBlue
logger.level(LOG_LEVEL_HIGHLIGHT, no=31, color="<fg #00FF00>") # Green
logger.level(LOG_LEVEL_ALERT, no=29, color="<fg #f39c12>")     # (bright) orange
logger.remove()
logger.add(sys.stderr, level="INFO")

DEFAULT_USER_PROMPT = "What color is the sky?"

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

# Note: this would be the method we would expand to format our own module hierarchy with:
def format_module_hierarchy(name:str, module:torch.nn.Module, indent=0) -> str:
    output = f"\n{'  ' * indent}({name}) {module.__class__.__name__} ({model.extra_repr()})"
    for name, child in module.named_children():
        output += format_module_hierarchy(name, child, indent + 1)
    return output

def getShape(t) -> Tuple[bool, str]:
    return (True, str(t.shape)) if isinstance(t, torch.Tensor) else (False, "NONE")

def format_bytes_to_gb(byte_count):
    gb_size = byte_count / (1024 ** 3)
    return f"{gb_size:.2f} GB"

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
            elif type(tensor) is list: #  built-in
                logger.log(LOG_LEVEL_ALERT, f"Unexpected module data type: {type(tensor)} expected 'torch.Tensor'")
                logger.log(LOG_LEVEL_ALERT, f"Converting to a torch.Tensor type...")
                # convert to tensor and output (recurse)
                actualTensor = torch.tensor([tensor])
                print_torch_tensors(hook_name, level, module_name, module_class_name, actualTensor, indent)
            else:
                logger.log(LOG_LEVEL_ALERT, f"Unknown module data type: {type(tensor)} expected 'torch.Tensor'")
                msg += f"\n[{i}]:{tensor}"
                logger.log(LOG_LEVEL_ALERT, f"{msg}")
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

def load_model(model_path: pathlib.Path, model_type: str):
    
    if model_type == 'text':
        return AutoModelForCausalLM.from_pretrained(local_path, low_cpu_mem_usage=True)
    elif model_type == 'vision':
        return AutoModelForVision2Seq.from_pretrained(local_path, low_cpu_mem_usage=True)
    else:
        raise ValueError(f"--model-type {model_type} does not exist")



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('-m', '--model-path', type=pathlib.Path, help='path to local HF model repo.')
        group.add_argument('-c', '--config-path', type=pathlib.Path, help='path to local HF model config file (initializes model with no pretrained data).')
        parser.add_argument('-p', '--prompt', required=False, default=DEFAULT_USER_PROMPT, help='Optional prompt text on model.generate()')
        parser.add_argument('--hook-pre-forward', default=False, action='store_true', help='Enable pre-forward hook on modules.')
        parser.add_argument('--module-hierarchy', default=True, action='store_true', help='Show module class hierarchy')
        parser.add_argument('--filter-class', type=str, nargs='+', required=False, help='Include only modules whose class name includes the substrings provided.')
        parser.add_argument('--filter-name', type=str, nargs='+', required=False, help='Include only modules whose name includes the substrings provided.')
        parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose output')
        parser.add_argument('--debug', default=False, action='store_true', help='Enable debug output')
        parser.add_argument('--trace', default=False, action='store_true', help='Enable trace output')
        parser.add_argument('--pause', default=False, action='store_true', help='Pause between forward hooks')
        parser.add_argument('--truncate-threshold', default=10, type=int, help='truncate tensor value printout if > threshold')
        parser.add_argument('--model-type', default='text', type=str, help='enter the model type, ex. vision, text')
        args = parser.parse_args()

        if args.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        if args.debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

        # allow cmd. line config of print opts.
        threshold = args.truncate_threshold
        torch.set_printoptions(threshold=threshold, linewidth=200, edgeitems=3) # TODO: others: precision=4, sci_mode=False

        # store flags for use in hook callbacks
        flags:DebugFlags = DebugFlags(args.debug, args.trace, args.pause)
        logger.trace(f"flags: {flags}")

        if args.model_path:
            local_path = str(args.model_path)
            if not args.model_path.exists():
                raise ValueError(f"--model-path {local_path} does not exist")
            if not args.model_path.is_dir():
                raise ValueError(f"--model-path {local_path} is not a model repo. (directory)")

            # load the model
            model = load_model(local_path, args.model_type)

        elif args.config_path:
            local_path = str(args.config_path)
            if not args.config_path.exists():
                raise ValueError(f"--config-path {local_path} does not exist")
            if not args.config_path.is_dir():
                raise ValueError(f"--config-path {local_path} is not a model repo. (directory)")

            config =  AutoConfig.from_pretrained(args.config_path, local_files_only=True)
            architectures = config.architectures
            if len(architectures) > 0:
                model_class_name = architectures[0]
                print(f"Using architectures: {architectures}: {model_class_name}")
                # model_class = get_class_from_module(model_class_name)
                model = AutoModelForCausalLM.from_config(config)
                model._init_weights(config)
                print(f"model: {model}")

        # Register "hooks" matching the following filters:
        if args.filter_class:
            logger.info(f"Registering hooks: for module_class: {args.filter_class}")

        if args.filter_name:
            logger.info(f"Registering hooks for module_names: {args.filter_name}")

        # Useful model information
        mem_size = model.get_memory_footprint()
        logger.log(LOG_LEVEL_SUMMARY, f"Model memory footprint: {format_bytes_to_gb(mem_size)}")

        # Only print the class hierarchy (once) using the top-level model, if requested
        if args.module_hierarchy:
            logger.log(LOG_LEVEL_SUMMARY, f"Module hierarchy (<name>) <class> :\n{str(model)}")
            out = format_module_hierarchy("class", model)
            logger.log(LOG_LEVEL_SUMMARY, f"\n{out}")

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
