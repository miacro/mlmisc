import os
import glob
import importlib


def modules():
    dirname = os.path.dirname(
        os.path.realpath(os.path.expanduser(os.path.expandvars(__file__))))
    names = [
        os.path.basename(filename)[:-3]
        for filename in glob.glob(os.path.join(dirname, "*.py"))
        if os.path.basename(filename) not in ("__init__.py", )
    ]
    modules = {
        name: importlib.import_module(
            ".annotations.{}".format(name), package="mlmisc")
        for name in names
    }
    return modules
