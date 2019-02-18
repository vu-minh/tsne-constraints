

+ Create virtual environment
```
python3 -m venv dash-tsne-const
```

+ Activate
```
source dash-tsne-const/bin/activate
```

+ Install all dependences
```
pip install --upgrade pip
pip install -r requirement.txt
```

Check installed packages with `pip list`

Dash app in `dash-app` folder.
The demo scrip is in `app-demo.py`, which is run by `python app-demo.py`.
Run with `gunicorn`: `gunicorn --bind 0.0.0.0:8050 app-demo:server`, in which loopback address allows using ip address in url, e.g., 'http://138.48.33.85:8050/'.
`app-demo` is main script, `server` is server instant in the `app-demo.py` script.
