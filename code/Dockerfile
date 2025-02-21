FROM python:3.12.9-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
