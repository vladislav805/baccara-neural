# Baccara prediction model
## Requirements
- Python 3.7.x;
- TensorFlow 2.3.1. 

## Installation and usage
-   Create virtual env
    ```shell
    python3 -m venv env
    ```
    and activate
    ```shell
    source venv/bin/activate
    ```
-   Install dependencies
    ```shell
    pip install -r requirements.txt
    ```
-   Train model
    ```shell
    python main.py train
    ```
-   Test on some data (checkout in code)
    ```shell
    python main.py predict  # one test
    python main.py masstest  # multiple test 
    ```
-   Run Telegram bot
    ```shell
    TG_TOKEN=123456:abcdef python main.py bot
    ```
