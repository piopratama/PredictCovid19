from flask import Flask

from dashboard.dashboard import dashboard
from fold_validation.fold_validation import fold_validation
from result.result import result
from detail.detail import detail
from user_testing.user_testing import user_testing

app = Flask(__name__)

app.secret_key = "my_secret_key_is_here"

app.config['UPLOAD_FOLDER'] = "./static/dataset"

app.register_blueprint(dashboard, url_prefix='/')
app.register_blueprint(fold_validation, url_prefix='/fold_validation')
app.register_blueprint(result, url_prefix='/result')
app.register_blueprint(detail, url_prefix='/detail')
app.register_blueprint(user_testing, url_prefix='/user_testing')

if __name__=='__main__':
    app.run()