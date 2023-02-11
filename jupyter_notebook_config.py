import os
from notebook.auth import passwd

# Note this is NOT a secure password. I include this
# just for ease of use. Please change this to be secure if
# you are running a public facing service
my_password = os.environ["JUPYTER_PASSWORD"]
hashed_password = passwd(passphrase=my_password, algorithm='sha256')

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.allow_password_change = True

c.NotebookApp.password = hashed_password
c.NotebookApp.open_browser = False

c.NotebookApp.port = 8888