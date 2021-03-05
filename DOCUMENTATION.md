# Documentation

## Viewing

### Online

An online version of this project's documentation is available at [paddy.pfiers.net](https://paddy.pfiers.net).

### Locally

1. Checkout the docs branch

```
git checkout docs
```


2. Open docs (method 1)

Open [_build/html/index.html](_build/html/index.html) in your browser.


**OR**

2. Open docs (method 2)

With [python 3.x](https://python.org) installed, run:
```
python3 -m http.server --directory "_build/html/"
```
And open [0.0.0.0:8000](http://0.0.0.0:8000/) in your browser.

## Building

1. Create an virtual enviroment.

```
python3 -m venv env
```


2. Activate the enviroment (depends on shell).

```
source env/bin/activate
```

3. Install requirements

```
pip install -r requirements
```

4. Build documentation

```
jupyter-book build .
```

To view the docs, see [Viewing > Locally](#locally)
