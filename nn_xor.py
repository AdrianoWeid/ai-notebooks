from math import exp
import random

# XOR-Trainingdaten
training_data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

# Gewichte und Biases initialisieren
weights_input_hidden = [random.uniform(-1, 1) for _ in range(4)]
weights_hidden_output = [random.uniform(-1, 1) for _ in range(2)]
bias_hidden = [random.uniform(-1, 1) for _ in range(2)]
bias_output = random.uniform(-1, 1)

# Sigmoid-Funktion
def sigmoid(x):
    return 1 / (1 + exp(-x))

# Verlustfunktion (Mean Squared Error)
def mse_loss(output, target):
    return 0.5 * (output - target) ** 2

# Vorwärtspropagation
def forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer = []
    for i in range(2):
        weighted_sum = inputs[0] * weights_input_hidden[i * 2] + inputs[1] * weights_input_hidden[i * 2 + 1] + bias_hidden[i]
        hidden_layer.append(sigmoid(weighted_sum))

    output = sigmoid(hidden_layer[0] * weights_hidden_output[0] + hidden_layer[1] * weights_hidden_output[1] + bias_output)
    return hidden_layer, output

# Rückwärtspropagation
def backpropagation(inputs, hidden_layer, output, target, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate=0.1):
    output_error = output - target
    output_delta = output_error * output * (1 - output)

    hidden_errors = []
    for i in range(2):
        hidden_error = output_delta * weights_hidden_output[i]
        hidden_delta = hidden_error * hidden_layer[i] * (1 - hidden_layer[i])
        hidden_errors.append(hidden_delta)

    # Update Gewichte und Biases (Hidden Layer)
    for i in range(2):
        weights_input_hidden[i * 2] -= learning_rate * hidden_errors[i] * inputs[0]
        weights_input_hidden[i * 2 + 1] -= learning_rate * hidden_errors[i] * inputs[1]
        bias_hidden[i] -= learning_rate * hidden_errors[i]

    # Update Gewichte und Biases (Output Layer)
    for i in range(2):
        weights_hidden_output[i] -= learning_rate * output_delta * hidden_layer[i]
    bias_output -= learning_rate * output_delta

# Trainingsfunktion
def train_nn(training_data, epochs=100000):
    global bias_output  # Da der Bias in der Rückwärtspropagation geändert wird
    for epoch in range(epochs):
        total_loss = 0
        for data in training_data:
            inputs = data[:2]
            target = data[2]

            # Vorwärtspropagation
            hidden_layer, output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

            # Verlust berechnen
            loss = mse_loss(output, target)
            total_loss += loss

            # Rückwärtspropagation
            backpropagation(inputs, hidden_layer, output, target, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

        # Verlust alle 1000 Epochen anzeigen
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss}')

# Testen des neuronalen Netzwerks
def main():
    train_nn(training_data)

    # Testen des neuronalen Netzwerks
    for data in training_data:
        inputs = data[:2]
        _, output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        print(f'Input: {inputs}, Predicted Output: {round(output)}, Validation Output: {data[2]}')

if __name__ == "__main__":
    main()
