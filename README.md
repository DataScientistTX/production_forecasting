# Production Forecasting Project

## Description
This project is designed for oil and gas production forecasting. It processes well data, calculates well characteristics, trains and evaluates multiple forecasting models, and generates various visualizations to aid in production analysis and prediction.

## Table of Contents
- [Production Forecasting Project](#production-forecasting-project)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Contributing](#contributing)
  - [License](#license)

## Installation
To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/production_forecasting.git
   cd production_forecasting
   ```

2. Install the required dependencies (make sure you have Python installed):
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the main analysis script:

```bash
python main.py
```

The script will prompt you to choose whether to use a subset of wells for faster processing. 

## Features
- Data loading and preprocessing
- Well characteristic calculation
- Data filtering and processing
- Multiple visualization types:
  - Oil production plots
  - Top 5 wells analysis
  - Cumulative production
  - Total production (oil and gas)
  - Producing wells over time
  - Gas-Oil Ratio (GOR) analysis
- Model training and evaluation
- Oil production prediction for specific wells

## Project Structure
```
production_forecasting/
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
│
├── data/
│   └── raw/
│       └── test.csv
│
├── outputs/
│   ├── figures/
│   └── model_cache/
│
└── main.py
```

## Dependencies
This project requires several Python libraries, including:
- pandas
- numpy
- matplotlib
- scikit-learn
- (Add any other specific libraries used in your project)

A complete list of dependencies can be found in the `requirements.txt` file.

## Contributing
Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).