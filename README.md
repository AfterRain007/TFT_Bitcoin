# Bitcoin forecasting using Temporal Fusion Transofrmers
Forecasting Bitcoin price using its Volume, Daily Sentiment, and Google Trends. Three models are compared against each other:
1. Temporal Fusion Transformers
2. Long-Short Term Memory
3. Gated Recurrent Networks.

## Dependencies
* Python		3.10.0.rc2
* Optuna		3.6.1
* Torch			2.3.0
* Numpy 		1.26.4
* Darts 		0.29.0
* Scikit-Learn		1.4.2
* Pandas 		2.2.2
* Pytorch_lightning	2.2.4
* Torchmetrics 		1.4.0
* Matplotlib		3.8.4

## Project Structure
* data: Data used
* res: Output (result) of the program
* util: Coding utilities used in the program
* main.py: Main script for running the whole program

## Credit
Credit where credit's due, thank you for all!<br>
[Main Reference](https://www.mdpi.com/2571-9394/5/1/10) by Kate Murray et al.<br>
[Main Refrence Source Code](https://github.com/katemurraay/tsa_crt/tree/kmm4_branch) by Kate Murray