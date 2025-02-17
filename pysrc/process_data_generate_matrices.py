import os
import sys

import numpy as np


def read_input_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]


def get_unique_characters(data):
    unique_small = set()
    unique_capital = set()
    for idx, line in enumerate(data):
        try:
            small, capital, _ = line
        except Exception as e:
            print(idx, line, e)
        finally:
            unique_small.update(small.split(","))
            unique_capital.update(capital.split(","))
    return sorted(unique_small), sorted(unique_capital)


def one_hot_encode(char_list, unique_chars):
    encoding = np.zeros(len(unique_chars), dtype=int)
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    for char in char_list:
        encoding[char_to_index[char]] = 1
    return encoding


def process_data(data, unique_small, unique_capital):
    processed_data = []
    final_small_matrix = None
    final_capital_matrix = None

    for line in data:
        try:
            small, capital, _ = line
        except Exception as e:
            print(line, e)
        finally:
            small_list = small.split(",")
            capital_list = capital.split(",")
            small_matrix = one_hot_encode(small_list, unique_small)
            capital_matrix = one_hot_encode(capital_list, unique_capital)
            final_small_matrix = small_matrix
            final_capital_matrix = capital_matrix
            small_flat = ",".join(map(str, small_matrix))
            capital_flat = ",".join(map(str, capital_matrix))
            processed_data.append(f"{small_flat} {capital_flat}")

    if final_small_matrix is not None and final_capital_matrix is not None:
        print(f"Final Small matrix shape: {final_small_matrix.shape}")
        print(f"Final Capital matrix shape: {final_capital_matrix.shape}")

    return processed_data


def write_output_file(output_path, processed_data):
    with open(output_path, "w") as file:
        file.write("\n".join(processed_data))


def write_keys_file(output_path, unique_chars):
    with open(output_path, "w") as file:
        file.write(",".join(unique_chars))


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_data_generate_matrices.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    data = read_input_file(input_file)
    unique_small, unique_capital = get_unique_characters(data)
    processed_data = process_data(data, unique_small, unique_capital)

    output_file = "-".join(input_file.split("-")[:-1]) + "-matrices.txt"
    write_output_file(output_file, processed_data)

    small_keys_file = "-".join(input_file.split("-")[:-1]) + "-small-keys.txt"
    capital_keys_file = "-".join(input_file.split("-")[:-1]) + "-capital-keys.txt"
    write_keys_file(small_keys_file, unique_small)
    write_keys_file(capital_keys_file, unique_capital)


if __name__ == "__main__":
    main()
