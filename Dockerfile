FROM python:3.11.2
RUN set -ex && mkdir /en_fr_neural_machine_translation
WORKDIR /en_fr_neural_machine_translation
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the files
COPY model/ ./model
COPY . ./
EXPOSE 8000
ENV PYTHONPATH /en_fr_neural_machine_translation
CMD python /en_fr_neural_machine_translation/app.py