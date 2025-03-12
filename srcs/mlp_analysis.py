#!/usr/bin/env python
#
# Main file to run the analysis of the MLP baseline
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/21
# Modified Date: 2025/02/05
# -----------------------------------------------------------------------------

import argparse
import inspect
import json
import os
import sys

# Package modules
current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel1 = os.path.dirname(current_path)
dirlevel0 = os.path.dirname(dirlevel1)

print(dirlevel0)

sys.path.insert(0, dirlevel0)

#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True, help="Please provide a config.json file"
    )

    args = parser.parse_args()

    print(args.config)
    with open(args.config) as f:
        config = json.load(f)

    config_id = 1
    for n_layers in [1, 2, 3, 5, 7, 10]:
        config["net_params"]["num_layers"] = n_layers

        for n_neurons in [4, 8, 16, 32, 64, 128]:
            config["net_params"]["hidden_dim"] = n_neurons

            outfile = args.config
            outfile = outfile.replace("2.0", "2.{:d}".format(config_id))

            print(outfile)

            with open(outfile, "w") as fh:
                json.dump(config, fh, indent=True)
            fh.close()

            config_id += 1
