FROM pytorch/pytorch
ADD . .
ADD .data/* .data/
ADD .models/* .models/
RUN pip install -r /requirements.txt
