# for running app in wsgi mode
# e.g. $ gunicorn dash-app.wsgi

from server import server as application
