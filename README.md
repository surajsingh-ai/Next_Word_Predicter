# Next Word Prediction App

A modern, interactive web application that predicts the next word in a sentence using an LSTM neural network.

## Features

- 🧠 LSTM-based word prediction model
- 🎨 Modern, responsive UI with gradient styling
- ⚡ Real-time predictions as you type
- 📊 Confidence scores for predictions
- 🔧 Adjustable number of predictions (1-5)
- 💫 Smooth animations and loading indicators

## Technologies Used

- **Frontend:** Streamlit
- **Backend:** TensorFlow/Keras (LSTM model)
- **Data Processing:** NumPy, Pickle
- **Styling:** Custom CSS

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/next-word-prediction.git
   cd next-word-prediction
   ```

2. Install dependencies:
   ```bash
   pip install streamlit tensorflow numpy
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser to `http://localhost:8501`

## Usage

1. Type a sentence in the input field
2. Watch as the AI predicts the next words in real-time
3. View confidence scores for each prediction
4. Adjust the number of predictions in the sidebar

## Model Details

- **Architecture:** Long Short-Term Memory (LSTM)
- **Training Data:** Text corpus (preprocessed)
- **Max Sequence Length:** Configurable
- **Vocabulary Size:** Based on tokenizer

## Contributing

Feel free to fork this repository and submit pull requests!

## License

MIT License - see LICENSE file for details.