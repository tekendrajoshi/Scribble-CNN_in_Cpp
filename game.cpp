#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include"functions.h"
#include <sstream>
#include <fstream>
#include <algorithm>








Image convertGridToImage(bool pixels[28][28]) {
    Image image(28, std::vector<float>(28, 0.0f));
    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
            image[y][x] = pixels[y][x] ? .0f : 0.0f;
    return image;


}






// Constants
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 650;
const int PIXEL_SIZE = 4; // Size of each pixel in the grid
const int GRID_SIZE = 112; // Size of the MNIST grid (28x28 pixels)  line 41/42
const int BUTTON_HEIGHT = 20;
const int GRID_OFFSET_X = (WINDOW_WIDTH - GRID_SIZE * PIXEL_SIZE) / 2;
const int GRID_OFFSET_Y = (WINDOW_HEIGHT - GRID_SIZE * PIXEL_SIZE) / 2; //this tells where to put the grid on the screen for now it centers the drawing grid in the opened window

// Categories for the MNIST dataset
// These are the digits 0-9, which are the categories in the MNIST dataset

// MNIST Categories (digits 0-9)
const std::vector<std::string> CATEGORIES = {
    "airplane", "apple", "bicycle", "book", "car",
    "cat", "chair", "clock", "cloud", "cup"
};

struct Button {
    SDL_Rect rect;
    std::string label;
    bool visible;
    SDL_Color color;
};

enum AppState {
    WELCOME_SCREEN,
    DRAWING_SCREEN,
    RESULT_SCREEN
};

// Function to render text using SDL_ttf
void renderText(SDL_Renderer* renderer, TTF_Font* font, const std::string& text, 
                int x, int y, SDL_Color color) {
    SDL_Surface* surface = TTF_RenderText_Solid(font, text.c_str(), color);
    if (!surface) {
        std::cerr << "Failed to create text surface: " << TTF_GetError() << std::endl;
        return;
    }
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        std::cerr << "Failed to create texture from surface: " << SDL_GetError() << std::endl;
        SDL_FreeSurface(surface);
        return;
    }
    
    SDL_Rect dstRect = {x, y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dstRect);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

// Function to draw a grid of pixels
void drawGrid(SDL_Renderer* renderer, bool pixels[GRID_SIZE][GRID_SIZE]) {
    // Draw grid background
    SDL_SetRenderDrawColor(renderer, 240, 240, 240, 255);
    SDL_Rect gridRect = {GRID_OFFSET_X, GRID_OFFSET_Y, 
                         GRID_SIZE * PIXEL_SIZE, GRID_SIZE * PIXEL_SIZE};
    SDL_RenderFillRect(renderer, &gridRect);
    
    // Draw grid lines
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    for (int i = 0; i <= GRID_SIZE; i++) {
        // Horizontal lines
        SDL_RenderDrawLine(renderer, 
            GRID_OFFSET_X, GRID_OFFSET_Y + i * PIXEL_SIZE,
            GRID_OFFSET_X + GRID_SIZE * PIXEL_SIZE, GRID_OFFSET_Y + i * PIXEL_SIZE);
        // Vertical lines
        SDL_RenderDrawLine(renderer, 
            GRID_OFFSET_X + i * PIXEL_SIZE, GRID_OFFSET_Y,
            GRID_OFFSET_X + i * PIXEL_SIZE, GRID_OFFSET_Y + GRID_SIZE * PIXEL_SIZE);
    }
    
    // Draw filled pixels
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (pixels[y][x]) {
                SDL_Rect pixelRect = {
                    GRID_OFFSET_X + x * PIXEL_SIZE,
                    GRID_OFFSET_Y + y * PIXEL_SIZE,
                    PIXEL_SIZE, PIXEL_SIZE
                };
                SDL_RenderFillRect(renderer, &pixelRect);
            }
        }
    }
}

// Function to clear the drawing grid
void clearGrid(bool pixels[GRID_SIZE][GRID_SIZE]) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            pixels[y][x] = false;
        }
    }
}



int main() {












    ImageSet filters = load_filters("assets/filters.txt");
    std::vector<std::vector<float>> fc_weights = load_fc_weights("assets/fc_weights.txt");
    std::vector<float> fc_biases = load_fc_biases("assets/fc_biases.txt");











    // Initialize random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    if (TTF_Init() != 0) {
        std::cerr << "TTF initialization failed: " << TTF_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "QuickDraw - MNIST",
        SDL_WINDOWPOS_CENTERED, 
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, 
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    if(!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if(!renderer) {
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
        if(!renderer) {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    }

    
    // Load fonts

    TTF_Font* titleFont = TTF_OpenFont("assets/fonts/arialbd.ttf", 32);
    TTF_Font* normalFont = TTF_OpenFont("assets/fonts/arialbd.ttf", 24);
    TTF_Font* resultFont = TTF_OpenFont("assets/fonts/arialbd.ttf", 36);
    
    if (!titleFont || !normalFont || !resultFont) {
        std::cerr << "Failed to load fonts: " << TTF_GetError() << std::endl;
    
    }

    
    // Initialize drawing grid
    bool pixels[GRID_SIZE][GRID_SIZE] = { false };
    int selectedCategory = -1;
    std::string selectedCategoryLabel = "";
    AppState app_state = WELCOME_SCREEN;
    float current_probability = 0.0f;
    int score = 0;
    
    // Create buttons
    Button categoryButtons[CATEGORIES.size()];
    Button clearButton = { {WINDOW_WIDTH/2 - 150, WINDOW_HEIGHT - 50, 120, BUTTON_HEIGHT}, 
                         "CLEAR", true, {70, 130, 180, 255} };
    Button submitButton = { {WINDOW_WIDTH/2 + 30, WINDOW_HEIGHT - 50, 120, BUTTON_HEIGHT}, 
                          "SUBMIT", true, {70, 130, 180, 255} };
    Button backButton = { {WINDOW_WIDTH/2 - 60, WINDOW_HEIGHT - 50, 120, BUTTON_HEIGHT}, 
                        "BACK", true, {70, 130, 180, 255} };
    
    // Initialize category buttons
    int buttonY = 150;
    for (int i = 0; i < CATEGORIES.size(); i++) {
        categoryButtons[i] = { {WINDOW_WIDTH/2 - 100, buttonY, 200, BUTTON_HEIGHT}, 
                             CATEGORIES[i], true, {70, 130, 180, 255} };
        buttonY += BUTTON_HEIGHT + 10;
    }

    // Main loop
    bool running = true;
    bool drawing = false;
    bool erasing = false;
    SDL_Event event;

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
           if (event.type == SDL_QUIT) {
            running = false;
            }

            
            // Mouse button pressed
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);
                
                if (app_state == WELCOME_SCREEN) {
                    // Check category buttons
                    for (int i = 0; i < CATEGORIES.size(); i++) {
                        if (mouseX >= categoryButtons[i].rect.x && 
                            mouseX <= categoryButtons[i].rect.x + categoryButtons[i].rect.w &&
                            mouseY >= categoryButtons[i].rect.y && 
                            mouseY <= categoryButtons[i].rect.y + categoryButtons[i].rect.h) {
                            
                            selectedCategory = i;
                            selectedCategoryLabel = CATEGORIES[i];
                            app_state = DRAWING_SCREEN;
                            clearGrid(pixels);
                        }
                    }
                }
                else if (app_state == DRAWING_SCREEN) {
                    // Check clear button
                    if (mouseX >= clearButton.rect.x && 
                        mouseX <= clearButton.rect.x + clearButton.rect.w &&
                        mouseY >= clearButton.rect.y && 
                        mouseY <= clearButton.rect.y + clearButton.rect.h) {
                        
                        clearGrid(pixels);
                    }
                    // Check submit button
                    else if (mouseX >= submitButton.rect.x && 
                             mouseX <= submitButton.rect.x + submitButton.rect.w &&
                             mouseY >= submitButton.rect.y && 
                             mouseY <= submitButton.rect.y + submitButton.rect.h) {
                        
                        


// here it starts the CNN forward pass
                        
                        // Downsample the 112x112 grid to 28x28 for CNN input
                        bool small_pixels[28][28] = { false };
                        for (int y = 0; y < 28; ++y) {
                            for (int x = 0; x < 28; ++x) {
                                // Sample the center of each 4x4 block
                                int startY = y * (GRID_SIZE / 28);
                                int startX = x * (GRID_SIZE / 28);
                                bool filled = false;
                                // If any pixel in the block is set, mark as true
                                for (int dy = 0; dy < (GRID_SIZE / 28); ++dy) {
                                    for (int dx = 0; dx < (GRID_SIZE / 28); ++dx) {
                                        if (pixels[startY + dy][startX + dx]) {
                                            filled = true;
                                            break;
                                        }
                                    }
                                    if (filled) break;
                                }
                                small_pixels[y][x] = filled;
                            }
                        }
                        Image input_image = convertGridToImage(small_pixels);  // convert downsampled pixels to CNN input
                        ForwardResult result = forward_pass(input_image, filters, fc_weights, fc_biases);  // run CNN

                        int predicted_digit = argmax(result.probabilities);  // actual predicted digit (0–9)
                        current_probability = result.probabilities[selectedCategory];  // how confident the model is in the user's chosen digit

                        
                        // Calculate score (0-100 based on probability)
                        score = static_cast<int>(current_probability * 100);
                        app_state = RESULT_SCREEN;
                    }
                    // Check if drawing on grid
                    else if (mouseX >= GRID_OFFSET_X && mouseX < GRID_OFFSET_X + GRID_SIZE * PIXEL_SIZE &&
                             mouseY >= GRID_OFFSET_Y && mouseY < GRID_OFFSET_Y + GRID_SIZE * PIXEL_SIZE) {
                        
                        int gridX = (mouseX - GRID_OFFSET_X) / PIXEL_SIZE;
                        int gridY = (mouseY - GRID_OFFSET_Y) / PIXEL_SIZE;
                        
                        if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
                            pixels[gridY][gridX] = true;
                            drawing = true;
                        }
                    }
                }
                else if (app_state == RESULT_SCREEN) {
                    // Check back button
                    if (mouseX >= backButton.rect.x && 
                        mouseX <= backButton.rect.x + backButton.rect.w &&
                        mouseY >= backButton.rect.y && 
                        mouseY <= backButton.rect.y + backButton.rect.h) {
                        
                        app_state = WELCOME_SCREEN;
                        selectedCategory = -1;
                        clearGrid(pixels);
                    }
                }
            }
            
            // Mouse button released
            else if (event.type == SDL_MOUSEBUTTONUP) {
                drawing = false;
            }
            
            // Mouse motion while drawing
            else if (event.type == SDL_MOUSEMOTION && drawing) {
                if (app_state == DRAWING_SCREEN) {
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    
                    if (mouseX >= GRID_OFFSET_X && mouseX < GRID_OFFSET_X + GRID_SIZE * PIXEL_SIZE &&
                        mouseY >= GRID_OFFSET_Y && mouseY < GRID_OFFSET_Y + GRID_SIZE * PIXEL_SIZE) {
                        
                        int gridX = (mouseX - GRID_OFFSET_X) / PIXEL_SIZE;
                        int gridY = (mouseY - GRID_OFFSET_Y) / PIXEL_SIZE;
                        
                        if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
                            // Set current pixel and neighbors for smoother drawing
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int nx = gridX + dx;
                                    int ny = gridY + dy;
                                    if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                                        pixels[ny][nx] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Handle key presses
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
                else if (event.key.keysym.sym == SDLK_c && app_state == DRAWING_SCREEN) {
                    clearGrid(pixels);
                }
            }
        }

        // Rendering
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);

        if (app_state == WELCOME_SCREEN) {
            // Title
            renderText(renderer, titleFont, "Welcome to MNIST QuickDraw!", 
                      WINDOW_WIDTH/2 - 220, 50, {50, 50, 50});
            renderText(renderer, normalFont, "Select a digit to draw:", 
                      WINDOW_WIDTH/2 - 100, 100, {80, 80, 80});
            
            // Category buttons
            for (int i = 0; i < CATEGORIES.size(); i++) {
                SDL_SetRenderDrawColor(renderer, categoryButtons[i].color.r, 
                                      categoryButtons[i].color.g, 
                                      categoryButtons[i].color.b, 255);
                SDL_RenderFillRect(renderer, &categoryButtons[i].rect);
                
                // Draw button border
                SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
                SDL_RenderDrawRect(renderer, &categoryButtons[i].rect);
                
                // Calculate text position to center it in the button
                int textWidth, textHeight;
                TTF_SizeText(normalFont, categoryButtons[i].label.c_str(), &textWidth, &textHeight);
                int textX = categoryButtons[i].rect.x + (categoryButtons[i].rect.w - textWidth) / 2;
                int textY = categoryButtons[i].rect.y + (categoryButtons[i].rect.h - textHeight) / 2;
                
                renderText(renderer, normalFont, categoryButtons[i].label,
                          textX, textY, {255, 255, 255});
            }
            
            // Footer
            renderText(renderer, normalFont, "Draw digits and test your skills!",
                      WINDOW_WIDTH/2 - 160, WINDOW_HEIGHT - 80, {100, 100, 100});
        }
        else if (app_state == DRAWING_SCREEN) {
            // Title
            std::string title = "Drawing: " + selectedCategoryLabel;
            renderText(renderer, titleFont, title.c_str(), 
                      WINDOW_WIDTH/2 - 100, 10, {50, 50, 50});
            
            // Draw the grid
            drawGrid(renderer, pixels);
            
            // Draw buttons with text
            SDL_SetRenderDrawColor(renderer, clearButton.color.r, clearButton.color.g, clearButton.color.b, 255);
            SDL_RenderFillRect(renderer, &clearButton.rect);
            SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
            SDL_RenderDrawRect(renderer, &clearButton.rect);
            
            int textWidth, textHeight;
            TTF_SizeText(normalFont, clearButton.label.c_str(), &textWidth, &textHeight);
            int textX = clearButton.rect.x + (clearButton.rect.w - textWidth) / 2;
            int textY = clearButton.rect.y + (clearButton.rect.h - textHeight) / 2;
            renderText(renderer, normalFont, clearButton.label, textX, textY, {255, 255, 255});

            SDL_SetRenderDrawColor(renderer, submitButton.color.r, submitButton.color.g, submitButton.color.b, 255);
            SDL_RenderFillRect(renderer, &submitButton.rect);
            SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
            SDL_RenderDrawRect(renderer, &submitButton.rect);
            
            TTF_SizeText(normalFont, submitButton.label.c_str(), &textWidth, &textHeight);
            textX = submitButton.rect.x + (submitButton.rect.w - textWidth) / 2;
            textY = submitButton.rect.y + (submitButton.rect.h - textHeight) / 2;
            renderText(renderer, normalFont, submitButton.label, textX, textY, {255, 255, 255});
            
            // Instructions
            renderText(renderer, normalFont, "Draw your digit in the grid above",
                      WINDOW_WIDTH/2 - 150, WINDOW_HEIGHT - 100, {100, 100, 100});
        }
        else if (app_state == RESULT_SCREEN) {
            // Title
            renderText(renderer, titleFont, "Results", 
                      WINDOW_WIDTH/2 - 60, 50, {50, 50, 50});
            
            // Selected category
            std::string categoryText = "You drew: " + selectedCategoryLabel;
            renderText(renderer, normalFont, categoryText.c_str(), 
                      WINDOW_WIDTH/2 - 80, 120, {80, 80, 80});
            
            // Probability
            std::stringstream probStream;
            probStream << "Probability: " << std::fixed << std::setprecision(1) << (current_probability * 100) << "%";
            renderText(renderer, resultFont, probStream.str().c_str(), 
                      WINDOW_WIDTH/2 - 120, 200, {30, 120, 30});
            
            // Score
            std::stringstream scoreStream;
            scoreStream << "Score: " << score << "/100";
            renderText(renderer, resultFont, scoreStream.str().c_str(), 
                      WINDOW_WIDTH/2 - 80, 280, {30, 120, 30});
            
            // Feedback
            std::string feedback;
            if (score >= 85) {
                feedback = "Excellent! Perfect match!";
            } else if (score >= 70) {
                feedback = "Good job! Close match!";
            } else if (score >= 50) {
                feedback = "Not bad! Keep practicing!";
            } else {
                feedback = "Try again! You can do better!";
            }
            renderText(renderer, normalFont, feedback.c_str(), 
                      WINDOW_WIDTH/2 - 140, 360, {150, 30, 30});
            
            // Back button
            SDL_SetRenderDrawColor(renderer, backButton.color.r, backButton.color.g, backButton.color.b, 255);
            SDL_RenderFillRect(renderer, &backButton.rect);
            SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
            SDL_RenderDrawRect(renderer, &backButton.rect);
            
            int textWidth, textHeight;
            TTF_SizeText(normalFont, backButton.label.c_str(), &textWidth, &textHeight);
            int textX = backButton.rect.x + (backButton.rect.w - textWidth) / 2;
            int textY = backButton.rect.y + (backButton.rect.h - textHeight) / 2;
            renderText(renderer, normalFont, backButton.label, textX, textY, {255, 255, 255});
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16); // ~60 FPS
    }

    // Cleanup
    TTF_CloseFont(titleFont);
    TTF_CloseFont(normalFont);
    TTF_CloseFont(resultFont);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}



/* TO RUN 
g++ game.cpp functions.cpp -IC:/sdl2/SDL2-2.26.5/x86_64-w64-mingw32/include -IC:/SDL2_ttf-2.20.2/x86_64-w64-mingw32/include -LC:/sdl2/SDL2-2.26.5/x86_64-w64-mingw32/lib -LC:/SDL2_ttf-2.20.2/x86_64-w64-mingw32/lib -lSDL2 -lSDL2_ttf -o game.exe
*/
