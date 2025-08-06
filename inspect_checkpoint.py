import torch
import sys
from collections import OrderedDict

def print_nested_keys(d, indent=0):
    """
    Recursively prints keys of a nested dictionary.
    """
    if not hasattr(d, 'items'):
        print(f"Expected a dictionary-like object, but got {type(d)}")
        return

    for key, value in d.items():
        # Indent and print the key
        print('  ' * indent + str(key))
        
        # If the value is another dictionary, recurse
        if isinstance(value, (dict, OrderedDict)):
            print_nested_keys(value, indent + 1)
        # If it's a tensor, print its shape and dtype
        elif hasattr(value, 'shape'):
             print('  ' * (indent + 1) + f"--> shape: {value.shape}, dtype: {value.dtype}")
        # Otherwise, just print its type
        else:
             print('  ' * (indent + 1) + f"--> type: {type(value)}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '/root/swx/track2/Falcon/ckpts/resume-state84_pointnav_ada_res18.pth'
        print(f"Usage: python {sys.argv[0]} <path_to_pth_file>")
        print(f"No file path provided. Using default: {file_path}")

    try:
        # Load the checkpoint on CPU to avoid GPU memory issues
        print(f"Loading checkpoint from {file_path}...")
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        
        print(f"Successfully loaded checkpoint. Inspecting keys...\n")
        print_nested_keys(state_dict)

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred while loading or inspecting the file: {e}")
