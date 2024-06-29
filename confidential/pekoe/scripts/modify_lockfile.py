import argparse
import pathlib
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=str, help="path to lockfile")
    parser.add_argument("--print-ignored", action="store_true", help="print ignored lines")
    args = parser.parse_args()

    path = pathlib.Path(args.lock)
    if not path.exists():
        raise RuntimeError("Cannot access lockfile.")

    c = re.compile("^[a-zA-Z][-_.\w]*==")
    lock_out = ""
    ignored_lines = ""
    with open(path) as fr:
        for line in fr.readlines():
            if c.match(line):
                lock_out += line
            else:
                ignored_lines += line

    with open(path, "w") as fw:
        fw.write(lock_out)

    if args.print_ignored:
        print("Ignored lines:")
        print(ignored_lines)


if __name__ == "__main__":
    main()
