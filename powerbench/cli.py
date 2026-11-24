"""Installed command-line entry point."""


def main() -> None:
    """Run the forecasting benchmark command-line interface.

    Imports are delayed so package metadata and configuration objects remain usable
    in environments where a device-aware PyTorch stack has not yet been installed.

    Returns:
        None; the delegated command controls process output and exit status.
    """
    from experiments.forecasting import main as forecasting_main

    forecasting_main()
