# Simple 0D mechanics simulator

In this repo we implement simple 0D mechanics simulator.

## Install
First create python3 virtual environment and install required packages:
```bash
python3 -m venv venv
. venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run
To run the simulator, execute:
```bash
python3 main.py
```

## Jacobian computations
The script `symbolic_computations.py` contains code for deriving the expression for the jacobian of the system using `sympy`.


## License
MIT

## Author
- Henrik Finsberg
- Joakim Sundnes
