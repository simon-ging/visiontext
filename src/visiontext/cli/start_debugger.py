from visiontext.imports import *  # noqa


def main():
    print(f"before breakpoint()")
    breakpoint()
    print(f"after breakpoint()")


if __name__ == "__main__":
    main()
