
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <chrono>    // for seeding with time
#include "functions.h"
#include <iomanip>  // for std::setprecision
#include <cstdlib> // for rand(), RAND_MAX
#include <ctime>   // for seeding rand()
using namespace std;



// FUNCTIONS TO STORE THE TRAINED PARAMETERS IN A FILE SO THAT WE CAN THEM USE TO TRAIN THE IMAGE SENT FROM THE GAME APP.

void save_filters(const ImageSet& filters, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing filters." << std::endl;
        return;
    }

    for (const auto& filter : filters) {
        for (const auto& row : filter) {
            for (float val : row) {
                file << val << " ";
            }
            file << "\n";
        }
        file << "\n"; // Separate filters
    }

    file.close();
}





void save_fc_weights(const std::vector<std::vector<float>>& weights, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing fc weights." << std::endl;
        return;
    }

    for (const auto& row : weights) {
        for (float val : row) {
            file << val << " ";
        }
        file << "\n";
    }

    file.close();
}



void save_fc_biases(const std::vector<float>& biases, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing fc biases." << std::endl;
        return;
    }

    for (float val : biases) {
        file << val << " ";
    }

    file << "\n";
    file.close();
}








const int EPOCHS= 6; // Number of epochs for training
// Your typedefs or using aliases if any
using Image = vector<vector<float>>;
using ImageSet = vector<Image>;

float random_float() {
    return ((float)rand() / RAND_MAX) * 0.2f - 0.1f;  // [-0.1, 0.1];  // Random float between -1 and 1
}

Image random_filter() {
    Image filter(3, std::vector<float>(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            filter[i][j] = random_float();
    return filter;
}

std::vector<std::vector<float>> random_weights(int output_size, int input_size) {
    std::vector<std::vector<float>> weights(output_size, std::vector<float>(input_size));
    for (auto& row : weights)
        for (auto& val : row)
            val = random_float();
    return weights;
}

std::vector<float> random_biases(int size) {
    std::vector<float> biases(size);
    for (auto& b : biases)
        b = random_float();
    return biases;
}



int main()
{


// Load the dataset (assuming your data is in "mnist_style.csv")
    LabeledDataset dataset = load_labeled_images_from_csv("mnist_style.csv");
    
    std::cout << "Total images loaded: " << dataset.images.size() << "\n";
    std::cout << "Total labels loaded: " << dataset.labels.size() << "\n";
    
    // print one image
    for(int i=0; i<dataset.images[0].size(); i++)
    {
        for(int j=0; j<dataset.images[0][i].size(); j++)
        {
            std::cout << dataset.images[0][i][j] << " "; //print only 2 digits after decimal
            std::cout << std::fixed << std::setprecision(2) << dataset.images[0][i][j] << "\t"; //print only 2 digits after decimal
        }
        std::cout << "\n";
    }
    
    // You can now use dataset.images and dataset.labels for your CNN

// now to create a combined dataset of images and their corresponding labels
// we will create a vector of ImageSet and a vector of labels
// where each label corresponds to the category of the image (0 for cat, 1 for dog,.... etc.)




/*  
ImageSet all_images;
vector<int> labels;


for (auto& img : cat_images) {
    all_images.push_back(img);
    labels.push_back(0); // label 0 for cat
}

for (auto& img : dog_images) {
    all_images.push_back(img);
    labels.push_back(1); // label 1 for dog
} */




int num_filters = 5; // Number of filters for the convolution layer

    // Fix filters declaration
std::vector<Image> filters(num_filters); // Declare filters as a vector of Image
for (int i = 0; i < num_filters; ++i) 
{
    filters[i] = random_filter();
}

// Initialize random seed
srand(static_cast<unsigned int>(time(0)));   //C++'s random number generator (rand()) always produces the same sequence of numbers unless you seed it with something different

// Define the fully connected layer weights and biases
int num_classes = 10; // Number of output classes (e.g., cat and dog)
// pooled height and pooled width sould be 13 and 13.  csuse while passing 28*28 into convolution it become 26*26 the after passing into maxpol it become half
int pooled_height = 13; // After max pooling
int pooled_width = 13; // After max pooling
// Flattened size will be 13 * 13 * num_filters



int flattened_size = num_filters * pooled_height * pooled_width;
std::vector<std::vector<float>> fc_weights = random_weights(num_classes, flattened_size);
std::vector<float> fc_biases = random_biases(num_classes);



// Now we can perform the forward pass for each image in the dataset
// This will convolve each image with the filters, apply ReLU, flatten the output, and then pass it through the fully connected layer.
std::cout << "Starting forward pass for all images..." << std::endl;
// Perform forward pass for each image in the dataset

int correct = 0; // Counter for correct predictions
float total_loss = 0.0f; // Variable to accumulate loss
float learning_rate = 0.01; // Learning rate for the fully connected layer
int total = dataset.images.size();


// Removed local ForwardResult struct definition to use the one from functions.h





// Loop through each epoch
for(int i=0; i<EPOCHS; i++)
{
    


    std::cout << "Epoch " << i + 1 << ": Forward pass for all images..." << std::endl;

    for (int j = 0; j < dataset.images.size(); ++j)
    {

        ForwardResult result = forward_pass(dataset.images[j], filters, fc_weights, fc_biases);

        // ======= Loss + Accuracy =======
        float loss = cross_entropy(result.probabilities, dataset.labels[j]); 
        total_loss += loss;

        int pred = argmax(result.probabilities);
        if (pred == dataset.labels[j])
        {
            correct++;
        }

        // ======= Backward Pass (FC only) =======
        backward_pass_fc(result.flattened_input, result.probabilities, dataset.labels[j], fc_weights, fc_biases, learning_rate);

    }


    // Now call the functions to save the trained parameters to files
    save_filters(filters, "assets/filters.txt");
    save_fc_weights(fc_weights, "assets/fc_weights.txt");
    save_fc_biases(fc_biases, "assets/fc_biases.txt");





    float accuracy = (float)correct / total * 100.0f;
    std::cout << "Epoch " << i+ 1 << " - Accuracy: " << accuracy << "%\n"; // Print accuracy for the epoch
    std::cout << "Epoch " << i + 1 << " - Average Loss: " << total_loss / total << "\n"; // average loss for the epoch
    correct = 0; // Reset correct count for the next epoch
    total_loss = 0.0f; // Reset total loss for the next epoch  
}
}
