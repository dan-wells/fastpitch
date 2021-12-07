import argparse

def parse_hifigan_args(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help, allow_abbrev=False)
    # empty parser just to work with inference.load_and_setup_model()
    return parser
