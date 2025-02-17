import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Constants for hyperparameters
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE_GENERATOR = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002

# Parse command line arguments
parser = argparse.ArgumentParser(description="Load data file.")
parser.add_argument("--datafile", type=str, required=True, help="Path to the data file")
args = parser.parse_args()

# Load data from file
data = pd.read_csv(args.datafile, header=None, delimiter=" ").values

# Split the data into input and output matrices
input_data = []
output_data = []
for row in data:
    input_data.append([float(x) for x in row[0].split(",")])
    output_data.append([float(x) for x in row[1].split(",")])

input_data = torch.tensor(input_data, dtype=torch.float32)
output_data = torch.tensor(output_data, dtype=torch.float32)

# Print total records in input and output data
print(f"Total records in input data: {len(input_data)}")
print(f"Total records in output data: {len(output_data)}")

# Split the data into train, validation, and test sets (60-20-20 split)
train_size = int(0.6 * len(input_data))
val_size = int(0.2 * len(input_data))
test_size = len(input_data) - train_size - val_size

train_input, val_input, test_input = torch.split(
    input_data, [train_size, val_size, test_size]
)
train_output, val_output, test_output = torch.split(
    output_data, [train_size, val_size, test_size]
)

# Print total records in each split
print(f"Total records in train set: {len(train_input)}")
print(f"Total records in validation set: {len(val_input)}")
print(f"Total records in test set: {len(test_input)}")

# Determine dimensions
input_dim = input_data.shape[1]
output_dim = output_data.shape[1]

print(f"Input dimension: {input_dim}")
print(f"Output dimension: {output_dim}")

# Create DataLoader for training data with shuffling
train_dataset = TensorDataset(train_input, train_output)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Initialize models
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Loss and Optimizers
d_criterion = nn.BCELoss()
g_criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GENERATOR)
d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)


# Function to evaluate the model on the validation set
def evaluate(generator, discriminator, val_input, val_output):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        noise = val_input
        fake_data = generator(noise)
        real_labels = torch.ones(len(val_output), 1)
        fake_labels = torch.zeros(len(fake_data), 1)

        real_loss = d_criterion(discriminator(val_output), real_labels)
        fake_loss = d_criterion(discriminator(fake_data), fake_labels)
        d_loss = real_loss + fake_loss

        g_loss = g_criterion(
            discriminator(fake_data), real_labels
        )  # Trick discriminator

        # Calculate accuracy
        fake_data_rounded = torch.round(fake_data)
        correct = (fake_data_rounded == val_output).sum().item()
        total = val_output.numel()
        accuracy = correct / total * 100

    generator.train()
    discriminator.train()
    tqdm.write(f"Generator Accuracy: {accuracy:.2f}%")
    return d_loss.item(), g_loss.item()


# Training loop
for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    for real_data, real_labels in train_loader:
        real_batch_size = real_data.size(0)

        noise = torch.randn(real_batch_size, input_dim)
        fake_data = generator(noise)
        real_labels = torch.ones(real_data.size(0), 1)
        fake_labels = torch.zeros(fake_data.size(0), 1)

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = d_criterion(discriminator(real_data), real_labels)
        fake_loss = d_criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        g_loss = g_criterion(
            discriminator(fake_data), real_labels
        )  # Trick discriminator
        g_loss.backward()
        g_optimizer.step()

    val_d_loss, val_g_loss = evaluate(generator, discriminator, val_input, val_output)
    if True:
        tqdm.write(
            f"\rEpoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Val D Loss: {val_d_loss}, Val G Loss: {val_g_loss}",
        )

# Evaluate on test set
test_d_loss, test_g_loss = evaluate(generator, discriminator, test_input, test_output)
print(f"Test D Loss: {test_d_loss}, Test G Loss: {test_g_loss}")


# Generate 10 samples and compare with test output
generator.eval()
with torch.no_grad():
    noise = torch.randn(10, input_dim)
    generated_samples = generator(noise)
    for i in range(10):
        print(f"Generated Sample {i+1}: {generated_samples[i]}")
        print(f"Expected Output {i+1}: {test_output[i]}")
