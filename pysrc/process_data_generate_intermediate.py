import os
import sys


def process_data(input_file):
    data = {}

    with open(input_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            identifier = parts[0]
            small_alpha = parts[2][-1]
            capital_alpha = parts[3][-1]

            if identifier not in data:
                data[identifier] = {"small": set(), "capital": set()}

            if small_alpha.islower():
                data[identifier]["small"].add(small_alpha)
            data[identifier]["capital"].add(capital_alpha)
            base_name = os.path.basename(input_file)
            name_parts = base_name.split(".")
            name_parts[-2] = "-".join(name_parts[-2].split("-")[:-1] + ["intermediate"])
            output_file = ".".join(name_parts)
            output_file = os.path.join(os.path.dirname(input_file), output_file)

    with open(output_file, "w") as file:
        for identifier, values in data.items():
            small_alphas = ",".join(sorted(values["small"]))
            capital_alphas = ",".join(sorted(values["capital"]))
            file.write(f"{small_alphas} {capital_alphas} {identifier}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <path_to_txt_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_data(input_file)
