class McCullochPittsNeuron:
    def __init__(self, num_inputs, weights=None, threshold=0):
        """
        Initialize the McCulloch and Pitts neuron model.

        :param num_inputs: Number of input values the neuron will take.
        :param weights: List of weights associated with the inputs.
        :param threshold: Threshold for the activation function.
        """
        self.num_inputs = num_inputs
        if weights is None:
            self.weights = [1] * num_inputs  # Default weights of 1
        else:
            self.weights = weights

        self.threshold = threshold

    def activation_function(self, total_input):
        """
        Activation function that mimics the step function.

        :param total_input: The summed weighted input.
        :return: 1 if the input exceeds or equals the threshold, otherwise 0.
        """
        return 1 if total_input >= self.threshold else 0

    def process(self, inputs):
        """
        Process the inputs through the neuron and generate output.

        :param inputs: List of input values (0 or 1).
        :return: The output of the neuron after applying the activation function.
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        # Calculate the total weighted input
        total_input = sum(input_val * weight for input_val, weight in zip(inputs, self.weights))

        # Pass the total input through the activation function
        return self.activation_function(total_input)


# Example usage:
if __name__ == "__main__":
    # Create a neuron with 3 inputs and predefined weights and threshold
    neuron = McCullochPittsNeuron(num_inputs=3, weights=[1, 1, 1], threshold=2)

    # Define some input vectors
    inputs = [0, 1, 1]  # Example input (binary)

    # Process the inputs through the neuron
    output = neuron.process(inputs)

    print(f"Inputs: {inputs}, Output: {output}")