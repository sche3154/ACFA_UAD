import importlib

def find_processing_using_name(processing_name):
    """
    Import the module "processing/[processing_name]_processing.py".
    """
    processing_filename = "data.processing." + processing_name + "_processing"
    processinglib = importlib.import_module(processing_filename)
    processing = None
    target_processing_name = processing_name.replace('_', '') + 'processing'
    for name, cls in processinglib.__dict__.items():
        if name.lower() == target_processing_name.lower():
            processing = cls

    if processing is None:
        print("In %s.py, there should be a dataprocessing with class name that matches %s in lowercase." % (processing_filename, target_processing_name))
        exit(0)
        
    return processing

def create_prcoessing(processing_name, opt):
    """
    Create a network based on the flags given in the options
    """

    processing_class = find_processing_using_name(processing_name)
    processing = processing_class(opt)

    return processing