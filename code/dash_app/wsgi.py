# for running app in wsgi mode
# e.g. $ gunicorn dash-app.wsgi

from .app import server as application
