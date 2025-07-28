//needed functions
// convolve relu  maxpool flattern fully_connected softmax
#include "functions.h"
#include <fstream>
#include <sstream>
#include <vector>
#include<cstring>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <climits>





using namespace std;
// Your typedefs or using aliases if any
using Image = vector<vector<float>>;
using ImageSet = vector<Image>;

// Function to load images from a CSV file
// Each line in the CSV file represents a flattened 28x28 image with pixel values separated by commas.
// The function reads the file, normalizes the pixel values to [0, 1], and converts them into a 2D matrix format.
// The function returns a vector of images i.e. set of images.



// TO RESIZE IMAGE
Image resize_image(const Image& input, int new_height = 28, int new_width = 28) {
    int old_height = input.size();
    int old_width = input[0].size();
    
    Image output(new_height, std::vector<float>(new_width));

    float row_scale = static_cast<float>(old_height) / new_height;
    float col_scale = static_cast<float>(old_width) / new_width;

    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            float old_i = i * row_scale;
            float old_j = j * col_scale;

            int i0 = static_cast<int>(old_i);
            int j0 = static_cast<int>(old_j);
            int i1 = std::min(i0 + 1, old_height - 1);
            int j1 = std::min(j0 + 1, old_width - 1);

            float di = old_i - i0;
            float dj = old_j - j0;

            // Bilinear interpolation
            float top = (1 - dj) * input[i0][j0] + dj * input[i0][j1];
            float bottom = (1 - dj) * input[i1][j0] + dj * input[i1][j1];
            float value = (1 - di) * top + di * bottom;

            output[i][j] = value;
        }
    }

    return output;
}








LabeledDataset load_labeled_images_from_csv(const std::string& filename) {
    std::ifstream file(filename);
    LabeledDataset dataset;
    std::string line;
    int loaded_count = 0;
    int skipped_count = 0;

    const int WIDTH = 28;
    const int HEIGHT = 28;
    const int TOTAL_PIXELS = WIDTH * HEIGHT;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> pixels;

        try {
            // First value is the label
            getline(ss, value, ',');
            int label = std::stoi(value);
            
            // Read the remaining pixel values
            while (getline(ss, value, ',')) {
                pixels.push_back(std::stof(value));
            }

            if (pixels.size() != TOTAL_PIXELS) {
                throw std::runtime_error("Expected " + std::to_string(TOTAL_PIXELS) + 
                                       " pixels but got " + std::to_string(pixels.size()));
            }

            // Convert flat vector to 2D image
            Image img(HEIGHT, std::vector<float>(WIDTH));
            for (int i = 0; i < HEIGHT; ++i) {
                for (int j = 0; j < WIDTH; ++j) {
                    // Normalize pixel values to [0, 1]
                    img[i][j] = pixels[i * WIDTH + j] / 255.0f;
                }
            }

            dataset.images.push_back(img);
            dataset.labels.push_back(label);
            loaded_count++;

        } catch (const std::exception& e) {
            skipped_count++;
            std::cerr << "Skipping row: " << e.what() << std::endl;
        }
    }

    std::cout << "Loaded " << loaded_count << " images (" 
              << skipped_count << " skipped) from " << filename << "\n";
    return dataset;
}

// This function reads a CSV file containing image data, where each line represents a flattened image.
// It converts the pixel values into a 2D vector format (Image) and returns a vector of these images (ImageSet).


//CONVOLVE FUNCTION
Image convolve(const Image& image, const Image& filter) {
    int H = image.size();
    int W = image[0].size();
    int KH = filter.size();
    int KW = filter[0].size();

    int OH = H - KH + 1;
    int OW = W - KW + 1;

    Image output(OH, std::vector<float>(OW));

    for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) 
        {
            //aile samma eauta image ko eauta specific pixel choose hgarya xam i,j wala
            float sum = 0.0;   // tesko value chai 0 initialize garyo
            for (int ki = 0; ki < KH; ++ki) 
            {
                for (int kj = 0; kj < KW; ++kj) 
                {
                    sum += image[i + ki][j + kj] * filter[ki][kj];
                }
            }
            output[i][j] = sum;    // mathi initialize gareko sum ma output store garyo
        }
    }

    return output;
}





//RELU FUNCTION
Image relu(const Image& image) 
{
    Image output = image; // Create a copy of the input image
    // Apply ReLU activation function
    // ReLU: f(x) = max(0, x)
    // This replaces all negative values with 0 and keeps positive values unchanged.
    int H = image.size();
    int W = image[0].size();
    for(int i = 0; i < H; ++i)
    {
        for(int j = 0; j < W; ++j)
        {
            if(output[i][j] < 0) {
                output[i][j] = 0; // Set negative values to 0
            }
            // Positive values remain unchanged. No need to explicitly set them, as they are already positive
        }
    }
    return output; // Return the modified image with ReLU applied
    
}





//MAXPOOL FUNCTION
Image maxpool(const Image& image, int pool_size)  // pool size=2 for 2x2 pooling
{
    int H = image.size();
    int W = image[0].size();
    int OH = H / pool_size;
    int OW = W / pool_size;

    Image output(OH, std::vector<float>(OW));

    for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) {
            float max_val = -INFINITY;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    int r = i * pool_size + pi;
                    int c = j * pool_size + pj;  
                //Compute the actual row (r) and column (c) in the original image corresponding to the current pooling window cell.
                    max_val = std::max(max_val, image[r][c]);
                }
            }
            output[i][j] = max_val;
        }
    }

    return output;
}



//FLATTEN FUNCTION
std::vector<float> flatten(const Image& image) 
{
    std::vector<float> flat; // Create a flat vector to hold the values
    // Iterate through each row and each value in the row to flatten the 2D image into a 1D vector
    int H = image.size();
    int W = image[0].size();
    for(int i = 0; i < H; ++i)
    {
        for(int j = 0; j < W; ++j)
        {
            flat.push_back(image[i][j]); // Add each value to the flat vector}
        }
    }
    return flat;

}


// FULLY CONNECTED LAYER FUNXCTON
// This function takes a flattened input vector, weights matrix, and biases vector to compute the output of a fully connected layer.
// It performs the dot product of the input with the weights and adds the biases.
std::vector<float> fully_connected(const std::vector<float>& input,
                                   const std::vector<std::vector<float>>& weights,
                                   const std::vector<float>& biases) 
{
    int output_size = weights.size();       // number of neurons
    int input_size = input.size();          // flattened input size

    std::vector<float> output(output_size, 0.0f);    // Initialize output values to 0

    for (int i = 0; i < output_size; ++i) 
    {
        for (int j = 0; j < input_size; ++j) 
        {
            output[i] += weights[i][j] * input[j];  // Dot product of input and weights for each neuron
            // This computes the weighted sum of inputs for each neuron in the fully connected layer.
        }
        output[i] += biases[i]; // add bias
    }

    return output;  // This returns the vector of raw output values (also called logits) from the fully connected layer.
    
}




// Softmax function to convert logits to probabilities
std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    std::vector<float> exp_values(logits.size());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_values[i] = exp(logits[i] - max_logit);
        sum_exp += exp_values[i];
    }
    
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = exp_values[i] / sum_exp;
    }
    return probs;
}
// This function takes a vector of logits (raw output values from the fully connected layer) and converts them into probabilities using the softmax function.
// It first computes the exponentials of the logits, sums them up, and then normalizes each exponential by dividing it by the total sum to get probabilities.




// Forward pass function that processes a set of images through the CNN
// It applies convolution, ReLU, max pooling, flattening, and fully connected layers to compute class probabilities for each image.

// bahira bata pathauda bulk ma pathaune tara vitra ko each function ma pathauda chai one by one pathaune loop layera.

ForwardResult forward_pass(
    const Image& image,
    const ImageSet& filters,
    const std::vector<std::vector<float>>& fc_weights,
    const std::vector<float>& fc_biases
) 
{
    ForwardResult result;
    
    // 1. Apply all filters and collect conv outputs

    std::vector<Image> conv_outputs;  // variable to Store outputs for an image from convolution layer for all filter
    // Loop through each filter and apply convolution
    for (const auto& filter : filters) 
    {
        conv_outputs.push_back(convolve(image, filter));
    }

    // 2. ReLU for each filter output
    for (auto& conv : conv_outputs) 
    {   // eauta filter ko output lai ekchoti relu vitra pathaune ako result lai naya variable ma narakhi puranai conv_outputs vanne variable nai update garne
        conv = relu(conv);
    }




    // aile samma bahira ko loop (loop each image) bata choose vako image lai sabai filter ma pathayera ako feature maps lai relu ma pathayera ako output lai conv_outputs(set of feature maps for single image) ma store gareko xa.





    // 3. MaxPool (2x2) for each feature map
    std::vector<Image> pooled_outputs;     // variable to store pooled outputs for each feature maps of that specific image

    for (const auto& relu_out : conv_outputs) 
    {  //one by one sabai relu outputs lai maxpool ma pathaune
        pooled_outputs.push_back(maxpool(relu_out, 2));
    }

    // This applies max pooling to each feature map, reducing its size by half (2x2 pooling).
    // pooled_outputs now contains the max-pooled outputs for each feature map of the current image.
    // Each pooled output is a smaller image (e.g., 13x13 if the original was 26x26).



    // 4. Flatten all pooled outputs into a single 1D vector

    //pooled_outputs contains multiple 2D matrices → one for each filter applied to that single image.
    /*   for single image we have multiple pooled outputs i.e. equal to no of filters we used previously 
    so  this function flatten all the pooled outputs of that image and then join them end to end to get single 1D vector */


        // result.flattened input 1D variable to store flattened input for fully connected layer
    for (const auto& pooled : pooled_outputs) 
    {  // Loop through each pooled output and flatten it

        std::vector<float> flat = flatten(pooled);
        result.flattened_input.insert(result.flattened_input.end(), flat.begin(), flat.end());   // Append all the flattened outputs to make single 1D vector
    } 
    // flattened_input now contains all the pixel values from all pooled outputs concatenated into a single vector.
    // This vector will be used as input to the fully connected layer.  ( size of that flattened input for single input will be  13 * 13)

    // 5. Fully Connected Layer
    std::vector<float> fc_output = fully_connected(result.flattened_input, fc_weights, fc_biases);

    // 6. Softmax
    result.probabilities = softmax(fc_output);




    return result; // returns a stuct containing the probabilities and flattened input for the fully connected layer.
    // The probabilities vector contains the class probabilities for the input image, and flattened_input contains the flattened feature map.
}




// FOR BACKPROPAGATION OF FC LAYER

//loss function for cross-entropy loss
// This function computes the cross-entropy loss between predicted probabilities and the true label.
// It uses a small epsilon value to avoid log(0) which can lead to numerical instability.

float cross_entropy(const std::vector<float>& probs, int label) {
    float epsilon = 1e-10f;
    // Ensure probability is clamped to avoid log(0)
    float p = std::max(probs[label], epsilon);
    return -log(p);
}



// argmax function to find the index of the maximum element in a vector

int argmax(const std::vector<float>& vec) 
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}
// This function finds the index of the maximum element in a vector.
// It returns the index of the class with the highest probability, which is used for prediction in classification tasks.


void backward_pass_fc(
    const std::vector<float>& flat_input,
    const std::vector<float>& probs,
    int label,
    std::vector<std::vector<float>>& fc_weights,
    std::vector<float>& fc_biases,
    float learning_rate
) 
{
    int num_classes = probs.size();
    std::vector<float> grad_softmax = probs; // Copy the softmax output

    // Subtract 1 from the correct class probability
    grad_softmax[label] -= 1.0f;

    // Gradient w.r.t. weights and biases
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < flat_input.size(); ++j) {
            fc_weights[i][j] -= learning_rate * grad_softmax[i] * flat_input[j];
        }
        fc_biases[i] -= learning_rate * grad_softmax[i];
    }
}
// This function performs the backward pass for the fully connected layer.
// It computes the gradients of the loss with respect to the weights and biases, and updates them using gradient descent.
// The gradients are calculated based on the difference between predicted probabilities and the true label.
// The learning rate controls how much to adjust the weights and biases during training.