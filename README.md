# Car Classifier

This is a web application built using **Flask** to classify images (using **AChorDS-LVQ**). The app predicts what type of car is in an image. In addition to the prediction, it also visualizes a heatmap, capturing the effect of each pixels on the model decision (in a form of heatmap).

## Features

- **Prediction**: Classify the type of the car in the input image (trained on 196 types of cars from the Standford Cars dataset).
- **Visualization**: Displays the influence of pixels on the model's prediction.
- **User-friendly Interface**: Upload an image (or a url of an image), and receive immediate predictions and visualizations.
  
## Requirements

To run this app locally, make sure you have the following installed:

- **Python 3.x**
- **pip** (Python package installer)

### Dependencies

Install the necessary Python packages by running:

```bash
pip install -r requirements.txt
```


The requirements.txt file includes the necessary dependencies like Flask, Matplotlib, and other libraries required for the app.


### Model Preparation

To use the app, you need to put a trained model in a directory `models`. Follow these steps:


## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/housing-eviction-predictor.git
cd housing-eviction-predictor
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

4. Open your browser and go to:

```arduino
http://127.0.0.1:5000/
```

5. Upload a text file with a legal case description, choose the number of influential words you'd like to display, and submit the form. The app will display the prediction and visualizations of the most influential words for each model.


## File Format

The input image should be a (.jpg) file. The app will process this image to predict what type of car is present in the image.

## Example

This is the visualization generated for an image:

![plot](./samples/example.png)


## Contributing

Feel free to fork this project and make improvements! If you find bugs or have suggestions for new features, please open an issue or create a pull request.
 
## References

Please consider citing the following work:

Mohammadi, M., Babai, M., & Wilkinson, M. H. F. (2024). Generalized Relevance Learning Grassmann Quantization. arXiv preprint arXiv:2403.09183.

## License

This project is open-source and available under the MIT License.






