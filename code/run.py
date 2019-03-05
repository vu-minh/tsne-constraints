# https://community.plot.ly/t/splitting-callback-definitions-in-multiple-files/10583/2
from dash_app.app import app

if __name__ == '__main__':
    app.run_server(debug=True)