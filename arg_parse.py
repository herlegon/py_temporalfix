from argparse import (
    ArgumentParser,
    Namespace,
    RawTextHelpFormatter,
)

from denc import (
    BoundedInteger,
    add_io_arguments,
    add_seek_arguments,
    add_debug_arguments,
)



def arg_parse() -> Namespace:
    parser = ArgumentParser(
        description="Python wrapper for vs_temporalfix script",
        formatter_class=RawTextHelpFormatter
    )
    add_io_arguments(parser)

    parser.add_argument(
        "-tr",
        "--t_radius",
        type=BoundedInteger(1, 10),
        metavar="[1..10]",
        default=6,
        required=False,
        help="""Temporal radius sets the number of frames to average over. Higher means more stable.
If you get blending/ghosting on small movements or blocky artifacts, reduce this.
\n"""
    )

    add_seek_arguments(parser)
    add_debug_arguments(parser)
    arguments: Namespace = parser.parse_args()

    return arguments
