"""Allow running the CLI via python -m CoreVital."""

import sys

from CoreVital.cli import main

if __name__ == "__main__":
    sys.exit(main())
