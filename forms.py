from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

class FaceForm(FlaskForm):
	source = FileField()
	destination = FileField()
	submit = SubmitField('Merge Faces')
