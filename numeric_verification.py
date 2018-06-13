#!/usr/bin/env python3

import sys

import backpropagation


def main(argv):
    return backpropagation.main(argv, backpropagation.numeric_verification)


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)
