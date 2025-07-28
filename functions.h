#ifndef TEKU_H
#define TEKU_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// Type aliases
typedef std::vector<std::vector<float>> Image;
typedef std::vector<Image> ImageSet;

struct ForwardResult;
// Forward pass result struct
struct ForwardResult {
    std::vector<float> probabilities;
    std::vector<float> flattened_input;
};

ForwardResult forward_pass(
    const std::vector<std::vector<float>>& image,
    const ImageSet& filters,
    const std::vector<std::vector<float>>& fc_weights,
    const std::vector<float>& fc_biases
);
struct LabeledDataset {
    ImageSet images;
    std::vector<int> labels;
};
// Data loading
LabeledDataset load_labeled_images_from_csv(const std::string& filename);

// CNN operations
Image convolve(const Image& image, const Image& filter);
Image relu(const Image& image);
Image maxpool(const Image& image, int pool_size);
std::vector<float> flatten(const Image& image);

// Fully connected layer
std::vector<float> fully_connected(const std::vector<float>& input,
                                   const std::vector<std::vector<float>>& weights,
                                   const std::vector<float>& biases);

// Softmax
std::vector<float> softmax(const std::vector<float>& logits);

// Loss
float cross_entropy(const std::vector<float>& probs, int label);

// Backpropagation
void backward_pass_fc(
    const std::vector<float>& flat_input,
    const std::vector<float>& probs,
    int label,
    std::vector<std::vector<float>>& fc_weights,
    std::vector<float>& fc_biases,
    float learning_rate
);

// Utility
int argmax(const std::vector<float>& vec);

#endif // TEKU_H
