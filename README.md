
## Requirements

* Python 3.6.x

## Install


        python -m venv _venv
        _venv/bin/pip install -U pip
        _venv/bin/pip install -r requirements.txt 


## Run

Simply call `_venv/bin/python gradient.py`. 
If you want to play around have look at the cli options using
`_venv/bin/python gradient.py --help`.

Always keep in mind, that you are fitting the sigmoid function. If you provide
multiple random points, it is likly that the error will **never** reach the
given threshold.
