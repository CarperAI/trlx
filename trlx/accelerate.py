import os
import sys
import json
from accelerate.commands import accelerate_cli


def main(hparams):
    # strip prepending arguments up until this script
    name = os.path.basename(__file__).replace(".py", "")
    for ix, arg in enumerate(sys.argv):
        if name in arg:
            sys.argv = sys.argv[ix:]
            break

    sys.argv.append(json.dumps(hparams))
    accelerate_cli.main()
