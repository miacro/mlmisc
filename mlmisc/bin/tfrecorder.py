from mlmisc import annotations
import pyconfigmanager as configmanager
import argparse
SCHEMA = {
    "command": "",
    "convert": {
        "annotype": {
            "$type": "str",
            "argparse": {
                "choices": [name for name, _ in annotations.modules().items()]
            }
        },
        "annofile": [],
        "metafile": [],
        "datafile": [],
        "label": "",
        "image": {
            "dir": "",
        },
        "rollover": False,
        "tfrecord": {
            "dir":
            "tfrecord",
            "batch_size":
            1000,
            "ratio": [0.99, 0.01, 0],
            "nameformats": [
                "{:0>8d}-trainset.tfrecord",
                "{:0>8d}-testset.tfrecord",
                "{:0>8d}-validationset.tfrecord",
            ],
        }
    },
    "inspect": {
        "filenames": [],
        "image": {
            "dir": "images"
        },
        "model": {
            "dir": "",
        }
    },
}


def main():
    config = configmanager.getconfig(schema=[
        SCHEMA, {
            key: value
            for key, value in configmanager.getschemas().items()
            if key in ("logging", "config")
        }
    ])

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="tfrecord dataset tools")
    subcommands = ["convert", "inspect"]
    config.update_values_by_argument_parser(
        parser=parser, subcommands=subcommands)
    if config.config.dump:
        config.dump_config(
            filename=config.config.dump, config_name="config.dump", exit=True)
    configmanager.logging.config(level=config.logging.verbosity)
    print(config)


if __name__ == "__main__":
    main()
