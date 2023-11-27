import numpy as np
from perceptron import Perceptron

def train_perceptron_for_logic_gate(logic_gate):
    if logic_gate == "XOR":
        return "Perceptron cannot model XOR gate as it is not linearly separable."

    training_data = {
        "AND": {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([0, 0, 0, 1])},
        "OR": {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([0, 1, 1, 1])},
        "NAND": {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([1, 1, 1, 0])},
    }

    perceptron = Perceptron()
    perceptron.train(training_data[logic_gate]["inputs"], training_data[logic_gate]["outputs"])
    return perceptron


# Logic gates to test
logic_gates = ["AND", "OR", "NAND", "XOR"]

for gate in logic_gates:
    print(f"Training the Perceptron for the {gate} logic gate...")

    if gate == "XOR":
        print("Perceptron cannot model XOR gate as it is not linearly separable.")
        continue

    perceptron = train_perceptron_for_logic_gate(gate)

    # Test the perceptron with different inputs
    for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        output = perceptron.predict(np.array(inputs))
        print(f"Testing {gate} with inputs {inputs}: {output}")

print("Training and testing for all gates complete.")