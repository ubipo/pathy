# Documentation

**TL;DR: [pathy.pfiers.net](https://pathy.pfiers.net)**

We use [Jupyter Book](https://jupyterbook.org/) (not to be confused with 
[Jupyter Notebooks](https://jupyter.org/)) to create HTML documentation 
from the .md, .py and .ipynb files in the repo (spread across the repo's 
different parts).

Any page of this rendered documentation can also be downloaded as pdf from
the relevant webpage.

![Find the "as pdf" button in the top-right corner](media/documentation-as-pdf.png)

By far the easiest way to view these docs is by browsing to
[pathy.pfiers.net](https://pathy.pfiers.net). However, you can also
view and/or build them locally:

## Viewing locally

1\. Checkout the docs branch

```
git checkout docs
```


2\. Open docs (method 1)

Open [_build/html/index.html](_build/html/index.html) in your browser.


**OR**

2\. Open docs (method 2)

With [python 3.x](https://python.org) installed, run:
```
python3 -m http.server --directory "_build/html/"
```
And open [0.0.0.0:8000](http://0.0.0.0:8000/) in your browser.

## Building

1\. Create an virtual enviroment.

```
python3 -m venv env
```


2\. Activate the enviroment (depends on shell).

```
source env/bin/activate
```


3\. Install requirements

```
pip install -r requirements.txt
```


4\. Update python path

```
export PYTHONPATH=$PYTHONPATH:$(pwd)/pathy_docutils/
```


5\. Build documentation

```
jupyter-book build .
```


To view the docs, see [Viewing locally](#viewing-locally)
